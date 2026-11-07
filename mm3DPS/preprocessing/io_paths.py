import os
from typing import Optional


def get_base_dir(seq_idx: int, frame_idx: int, out_root: str) -> str:
    return os.path.join(out_root, f"seq_{seq_idx}_frame_{frame_idx}")


def get_subdir(base_dir: str, name: str) -> str:
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def find_lowest_cascaded_config(base_dir: str) -> Optional[str]:
    cfg_dir = os.path.join(base_dir, "configs")
    if not os.path.isdir(cfg_dir):
        return None
    try:
        entries = os.listdir(cfg_dir)
        candidates: list[tuple[int, str]] = []
        for name in entries:
            if name.startswith("cascaded_frame_") and name.endswith(".json"):
                try:
                    idx = int(name[len("cascaded_frame_"):-len(".json")])
                    candidates.append((idx, os.path.join(cfg_dir, name)))
                except Exception:
                    continue
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    except Exception:
        return None


