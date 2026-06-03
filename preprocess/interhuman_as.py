import argparse
import json
import os
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.append("./")
import utils.rotation_conversions as geometry


SPLITS = ("train", "val", "test")
PERSON_KEYS = ("trans", "root_orient", "pose_body")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert InterHuman-AS pkl files to frozen ReGenNet H5 files."
    )
    parser.add_argument(
        "--data_root",
        default="dataset/interhuman",
        help="InterHuman root containing motions, split and annotations_interhuman.",
    )
    parser.add_argument(
        "--output_dir",
        default="dataset/interhuman/smpl/conditioned",
        help="Directory for interhuman_{split}.h5 and meta.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def read_split_ids(split_dir, split):
    with (split_dir / f"{split}.txt").open("r") as f:
        return [line.strip() for line in f if line.strip()]


def load_labels(label_path):
    with label_path.open("r") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def finite_array(value):
    array = np.asarray(value)
    return np.isfinite(array).all()


def validate_person(person, frames, person_name):
    expected_shapes = {
        "trans": (frames, 3),
        "root_orient": (frames, 3),
        "pose_body": (frames, 63),
    }
    for key in PERSON_KEYS:
        if key not in person:
            return False, f"missing_{person_name}_{key}"
        array = np.asarray(person[key])
        if array.shape != expected_shapes[key]:
            return False, f"invalid_shape_{person_name}_{key}:{array.shape}"
        if not finite_array(array):
            return False, f"non_finite_{person_name}_{key}"
    return True, "ok"


def load_valid_item(motion_dir, labels, sample_id):
    path = motion_dir / f"{sample_id}.pkl"
    if not path.exists():
        return None, "missing_motion_file"
    if sample_id not in labels:
        return None, "missing_actor_reactor_label"
    if labels[sample_id] not in (0, 1):
        return None, f"invalid_actor_reactor_label:{labels[sample_id]}"

    try:
        with path.open("rb") as f:
            item = pickle.load(f)
    except Exception as exc:
        return None, f"pickle_load_error:{type(exc).__name__}"

    frames = int(item.get("frames", 0))
    if frames <= 0:
        return None, "non_positive_frames"

    for person_name in ("person1", "person2"):
        if person_name not in item:
            return None, f"missing_{person_name}"
        ok, reason = validate_person(item[person_name], frames, person_name)
        if not ok:
            return None, reason

    return item, "ok"


def person_axis_angle(person):
    root = np.asarray(person["root_orient"], dtype=np.float32)
    body = np.asarray(person["pose_body"], dtype=np.float32)
    pad = np.zeros((root.shape[0], 6), dtype=np.float32)
    return np.concatenate([root, body, pad], axis=1).reshape(root.shape[0], 24, 3)


def axis_angle_to_rot6d(axis_angle):
    tensor = torch.from_numpy(axis_angle.astype(np.float32))
    with torch.no_grad():
        rot6d = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(tensor))
    return rot6d.cpu().numpy().astype(np.float32)


def ordered_people(item, label):
    person1, person2 = item["person1"], item["person2"]
    if label == 1:
        return person2, person1
    return person1, person2


def convert_item(item, label):
    actor, reactor = ordered_people(item, label)
    actor_rot6d = axis_angle_to_rot6d(person_axis_angle(actor))
    reactor_rot6d = axis_angle_to_rot6d(person_axis_angle(reactor))

    # 与在线 loader 一致，原因是双人相对位移需要共享同一个世界原点。
    origin = np.asarray(actor["trans"], dtype=np.float32)[0].copy()
    actor_trans = np.asarray(actor["trans"], dtype=np.float32) - origin
    reactor_trans = np.asarray(reactor["trans"], dtype=np.float32) - origin

    frames = actor_rot6d.shape[0]
    output = np.zeros((frames, 25, 12), dtype=np.float32)
    output[:, :24, 0:6] = actor_rot6d
    output[:, :24, 6:12] = reactor_rot6d
    output[:, 24, 0:3] = actor_trans
    output[:, 24, 6:9] = reactor_trans
    return output


def output_paths(output_dir):
    paths = {split: output_dir / f"interhuman_{split}.h5" for split in SPLITS}
    paths["meta"] = output_dir / "meta.json"
    return paths


def ensure_can_write(paths, overwrite):
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them: "
            + ", ".join(existing)
        )


def write_split(split, ids, motion_dir, labels, output_path):
    skipped = []
    shapes = {}
    written = 0
    with h5py.File(output_path, "w") as h5:
        for sample_id in ids:
            item, reason = load_valid_item(motion_dir, labels, sample_id)
            if item is None:
                skipped.append({"id": sample_id, "reason": reason})
                continue
            value = convert_item(item, labels[sample_id])
            if not np.isfinite(value).all():
                skipped.append({"id": sample_id, "reason": "converted_non_finite"})
                continue
            h5.create_dataset(sample_id, data=value, dtype="f4")
            shapes[sample_id] = list(value.shape)
            written += 1

    reason_counter = Counter(entry["reason"] for entry in skipped)
    return {
        "listed": len(ids),
        "written": written,
        "skipped": len(skipped),
        "skip_reasons": dict(reason_counter),
        "skipped_samples": skipped,
        "shapes": {
            "first_samples": dict(list(shapes.items())[:10]),
            "unique": sorted({tuple(shape) for shape in shapes.values()}),
        },
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    motion_dir = data_root / "motions"
    split_dir = data_root / "split"
    label_path = data_root / "annotations_interhuman" / "interhuman_label.json"

    paths = output_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_can_write(paths, args.overwrite)

    labels = load_labels(label_path)
    split_ids = {split: read_split_ids(split_dir, split) for split in SPLITS}

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "format": {
            "shape": "[T,25,12]",
            "rotation": "rot6d",
            "body_model": "smpl",
            "translation_slot": 24,
            "translation_origin": "actor frame 0",
            "channels": {
                "actor_rot6d": "[:, :24, 0:6]",
                "actor_translation": "[:, 24, 0:3]",
                "reactor_rot6d": "[:, :24, 6:12]",
                "reactor_translation": "[:, 24, 6:9]",
            },
        },
        "splits": {},
    }

    for split in SPLITS:
        print(f"Writing {split} -> {paths[split]}")
        meta["splits"][split] = write_split(
            split, split_ids[split], motion_dir, labels, paths[split]
        )

    with paths["meta"].open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(json.dumps(meta["splits"], indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
