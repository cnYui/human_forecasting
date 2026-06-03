import torch


def forecasting_collate(batch):
    not_none = [item for item in batch if item is not None]
    if len(not_none) == 0:
        raise ValueError("batch 为空")

    obs = torch.stack([item["obs"] for item in not_none], dim=0)
    target = torch.stack([item["target"] for item in not_none], dim=0)
    meta = []
    for item in not_none:
        meta.append(
            {
                "sample_id": item["sample_id"],
                "start": int(item["start"]),
                "length": int(item["length"]),
            }
        )

    return obs, target, meta
