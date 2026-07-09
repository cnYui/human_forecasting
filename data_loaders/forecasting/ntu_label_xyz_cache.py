import torch
from torch.utils.data import Dataset


class NTULabelXYZCacheDataset(Dataset):
    def __init__(self, cache_path, max_samples=-1):
        self.cache_path = cache_path
        data = torch.load(cache_path, map_location="cpu")
        required = ("obs_xyz", "target_xyz", "actions", "meta")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError("xyz cache 缺少字段: {}".format(missing))
        self.obs_xyz = data["obs_xyz"].float()
        self.target_xyz = data["target_xyz"].float()
        self.actions = data["actions"].long()
        self.meta = list(data["meta"])
        if self.actions.dim() == 1:
            self.actions = self.actions.view(-1, 1)
        count = int(self.obs_xyz.shape[0])
        if int(self.target_xyz.shape[0]) != count or int(self.actions.shape[0]) != count or len(self.meta) != count:
            raise ValueError("xyz cache 样本数不一致")
        if max_samples is not None and int(max_samples) > 0:
            limit = int(max_samples)
            self.obs_xyz = self.obs_xyz[:limit]
            self.target_xyz = self.target_xyz[:limit]
            self.actions = self.actions[:limit]
            self.meta = self.meta[:limit]

    def __len__(self):
        return int(self.obs_xyz.shape[0])

    def __getitem__(self, index):
        return {
            "obs_xyz": self.obs_xyz[index],
            "target_xyz": self.target_xyz[index],
            "action": self.actions[index],
            "meta": self.meta[index],
        }


def ntu_label_xyz_cache_collate(batch):
    not_none = [item for item in batch if item is not None]
    if len(not_none) == 0:
        raise ValueError("batch 为空")
    return {
        "obs_xyz": torch.stack([item["obs_xyz"] for item in not_none], dim=0).float(),
        "target_xyz": torch.stack([item["target_xyz"] for item in not_none], dim=0).float(),
        "action": torch.stack([item["action"] for item in not_none], dim=0).long(),
        "meta": [item["meta"] for item in not_none],
    }
