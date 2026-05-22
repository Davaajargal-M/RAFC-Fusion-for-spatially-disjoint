from pathlib import Path
from typing import Tuple, Dict, Optional

import json
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt


class HyperspectralDataset(Dataset):
    """Unified spatially disjoint dataset class for Houston2013, MUUFL, and Augsburg."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        patch_size: int = 7,
        data_root: Optional[str] = None,
        n_train_per_class: int = 7,
        n_val_per_class: int = 7,
        seed: int = 42,
        spatial_grid: Tuple[int, int] = (4, 4),
        train_block_ratio: float = 0.60,
        val_block_ratio: float = 0.00,
        max_split_tries: int = 50,
        use_spatial_buffer: bool = True,
        val_from_train_blocks: bool = True,
        split_cache_dir: Optional[str] = None,
        force_rebuild_split: bool = False,
    ):
        self.dataset_name = dataset_name.lower()
        self.split = split.lower()
        self.patch_size = int(patch_size)
        self.n_train_per_class = int(n_train_per_class)
        self.n_val_per_class = int(n_val_per_class)
        self.seed = int(seed)
        self.spatial_grid = tuple(spatial_grid)
        self.train_block_ratio = float(train_block_ratio)
        self.val_block_ratio = float(val_block_ratio)
        self.dev_block_ratio = self.train_block_ratio + self.val_block_ratio
        self.max_split_tries = int(max_split_tries)
        self.use_spatial_buffer = bool(use_spatial_buffer)
        self.val_from_train_blocks = bool(val_from_train_blocks)
        self.force_rebuild_split = bool(force_rebuild_split)

        dataset_lower = dataset_name.lower()

        if dataset_lower == "muufl":
            self.spatial_grid = (8, 8)
            train_block_ratio = self.train_block_ratio
            val_block_ratio = self.val_block_ratio
            dev_block_ratio = self.train_block_ratio + self.val_block_ratio
            self.max_split_tries = max(self.max_split_tries, 1000)
            self.use_spatial_buffer = False

        elif dataset_lower == "houston2013":
            self.spatial_grid = (6, 6)
            train_block_ratio = self.train_block_ratio
            val_block_ratio = self.val_block_ratio
            dev_block_ratio = self.train_block_ratio + self.val_block_ratio
            self.max_split_tries = max(self.max_split_tries, 1000)
            self.use_spatial_buffer = False

        elif dataset_lower == "augsburg":
            self.spatial_grid = (6, 6)
            train_block_ratio = self.train_block_ratio
            val_block_ratio = self.val_block_ratio
            dev_block_ratio = self.train_block_ratio + self.val_block_ratio
            self.max_split_tries = max(self.max_split_tries, 1000)
            self.use_spatial_buffer = False

        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: 'train', 'val', 'test'")

        if data_root is None:
            data_root = "D:/datasets/Houston, Trento, Muufl"
        self.data_root = data_root

        if split_cache_dir is None:
            split_cache_dir = "official_splits"
        self.split_cache_dir = Path(split_cache_dir)
        (self.split_cache_dir / self.dataset_name).mkdir(parents=True, exist_ok=True)

        self.data = self._load_dataset()
        self._create_or_load_spatial_split_indices()

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def _load_dataset(self) -> Dict:
        data = {}

        if self.dataset_name == "houston2013":
            base_path = Path(self.data_root) / "Houston2013" / "2"
            hsi_data = sio.loadmat(base_path / "houston_hsi.mat")
            lidar_data = sio.loadmat(base_path / "houston_lidar.mat")
            labels_data = sio.loadmat(base_path / "houston_gt.mat")

            data["hsi"] = hsi_data["houston_hsi"].astype(np.float32)
            data["lidar"] = lidar_data["houston_lidar"].astype(np.float32)
            data["labels"] = labels_data["houston_gt"].astype(np.int64)

        elif self.dataset_name == "muufl":
            base_path = Path(self.data_root) / "muufl"
            hsi_data = sio.loadmat(base_path / "MUUFL_hsi.mat")
            lidar_data = sio.loadmat(base_path / "MUUFL_lidar.mat")
            labels_data = sio.loadmat(base_path / "MUUFL_gt.mat")

            data["hsi"] = hsi_data["hsi"].astype(np.float32)
            data["lidar"] = lidar_data["lidar"].astype(np.float32)
            data["labels"] = labels_data["gt"].astype(np.int64)

        elif self.dataset_name == "augsburg":
            base_path = Path(self.data_root)
            hsi_data = sio.loadmat(base_path / "augsburg_hsi.mat")
            lidar_data = sio.loadmat(base_path / "augsburg_lidar.mat")
            labels_data = sio.loadmat(base_path / "augsburg_gt.mat")

            data["hsi"] = hsi_data["augsburg_hsi"].astype(np.float32)
            data["lidar"] = lidar_data["data_DSM"].astype(np.float32)
            data["labels"] = labels_data["augsburg_gt"].astype(np.int64)

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        data["hsi"] = self._normalize(data["hsi"])
        data["lidar"] = self._normalize(data["lidar"])
        return data

    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        data_min = data.min()
        data_max = data.max()
        if data_max > data_min:
            return (data - data_min) / (data_max - data_min)
        return data

    # ------------------------------------------------------------------
    # Spatial split construction - official splits
    # ------------------------------------------------------------------
    def _split_cache_path(self) -> Path:
        gh, gw = self.spatial_grid

        suffix = (
            f"split_seed{self.seed:03d}"
            f"_grid{gh}x{gw}"
            f"_tr{self.n_train_per_class}"
            f"_val{self.n_val_per_class}"
            f"_ps{self.patch_size}"
            f"_buffer{int(self.use_spatial_buffer)}"
            f"_valdev{int(self.val_from_train_blocks)}.npz"
        )

        return self.split_cache_dir / self.dataset_name / suffix

    def _create_or_load_spatial_split_indices(self):
        cache_path = self._split_cache_path()

        if cache_path.exists() and not self.force_rebuild_split:
            obj = np.load(cache_path, allow_pickle=True)
            self.train_indices = [tuple(x) for x in obj["train_indices"].tolist()]
            self.val_indices = [tuple(x) for x in obj["val_indices"].tolist()]
            self.test_indices = [tuple(x) for x in obj["test_indices"].tolist()]
            self.class_counts = obj["class_counts"].item()
            self.block_split_map = obj["block_split_map"]
        else:
            self._create_spatial_split_indices()
            np.savez_compressed(
                cache_path,
                train_indices=np.asarray(self.train_indices, dtype=np.int64),
                val_indices=np.asarray(self.val_indices, dtype=np.int64),
                test_indices=np.asarray(self.test_indices, dtype=np.int64),
                class_counts=self.class_counts,
                block_split_map=self.block_split_map,
            )
            print(f"[DEBUG] spatial_grid={self.spatial_grid}")
            print(f"[DEBUG] dev_block_ratio={getattr(self, 'dev_block_ratio', None)}")
            print(f"[DEBUG] train_block_ratio={self.train_block_ratio}")
            print(f"[DEBUG] val_block_ratio={self.val_block_ratio}")
            print(f"[DEBUG] val_from_train_blocks={self.val_from_train_blocks}")
            
            self._save_split_report(cache_path.with_suffix(".json"))

        if self.split == "train":
            self.indices = self.train_indices.copy()
        elif self.split == "val":
            self.indices = self.val_indices.copy()
        else:
            self.indices = self.test_indices.copy()

        rng = np.random.RandomState(self.seed)
        rng.shuffle(self.indices)

    def _create_spatial_split_indices(self):
        labels = self.data["labels"]
        h, w = labels.shape
        gh, gw = self.spatial_grid
        rng_master = np.random.RandomState(self.seed)

        block_id_map, block_coords = self._make_block_id_map(h, w, gh, gw)
        num_blocks = len(block_coords)

        required = self.n_train_per_class + self.n_val_per_class
        classes = sorted([int(c) for c in np.unique(labels) if int(c) > 0])

        best_split = None
        best_score = -1

        for attempt in range(self.max_split_tries):
            rng = np.random.RandomState(rng_master.randint(0, 2**31 - 1))
            block_ids = np.arange(num_blocks)
            rng.shuffle(block_ids)

            if self.val_from_train_blocks:
                dev_ratio = getattr(
                    self,
                    "dev_block_ratio",
                    self.train_block_ratio + self.val_block_ratio
                )

                n_dev_blocks = max(1, int(round(num_blocks * dev_ratio)))
                n_dev_blocks = min(n_dev_blocks, num_blocks - 1)

                train_blocks = set(block_ids[:n_dev_blocks].tolist())
                val_blocks = set()
                test_blocks = set(block_ids[n_dev_blocks:].tolist())

            split_map = np.full((gh, gw), fill_value=2, dtype=np.int64)  # 0=train, 1=val, 2=test
            for bid in train_blocks:
                br, bc = block_coords[bid]
                split_map[br, bc] = 0
            for bid in val_blocks:
                br, bc = block_coords[bid]
                split_map[br, bc] = 1

            train_pool, val_pool, test_pool = self._collect_split_pools(
                labels, block_id_map, train_blocks, val_blocks, test_blocks
            )

            if self.use_spatial_buffer:
                train_pool, val_pool, test_pool = self._apply_center_buffer_fast(
                    train_pool, val_pool, test_pool, radius=self.patch_size // 2, shape=labels.shape,)

            ok = True
            score = 0

            for c in classes:
                ntr = len(train_pool.get(c, []))
                nva = len(val_pool.get(c, []))
                nte = len(test_pool.get(c, []))

                if self.val_from_train_blocks:
                    dev_count = ntr
                    required_dev = self.n_train_per_class + self.n_val_per_class

                    score += (
                        min(dev_count, required_dev)
                        + min(nte, 50)
                    )

                    if dev_count < required_dev or nte < 1:
                        ok = False

                else:
                    score += (
                        min(ntr, self.n_train_per_class)
                        + min(nva, self.n_val_per_class)
                        + min(nte, 50)
                    )

                    if ntr < self.n_train_per_class or nva < self.n_val_per_class or nte < 1:
                        ok = False

            if score > best_score:
                best_score = score
                best_split = (train_pool, val_pool, test_pool, split_map.copy())

            if ok:
                best_split = (train_pool, val_pool, test_pool, split_map.copy())
                break

        if best_split is None:
            raise RuntimeError("Could not construct a spatial split.")

        train_pool, val_pool, test_pool, split_map = best_split
        self.block_split_map = split_map

        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        self.class_counts = {}

        rng = np.random.RandomState(self.seed)
        for c in classes:
            tr = train_pool.get(c, []).copy()
            va = val_pool.get(c, []).copy()
            te = test_pool.get(c, []).copy()

            rng.shuffle(tr)
            rng.shuffle(va)
            rng.shuffle(te)

            if self.val_from_train_blocks:
                dev = tr.copy()
                rng.shuffle(dev)

                required_dev = self.n_train_per_class + self.n_val_per_class

                if len(dev) < required_dev or len(te) < 1:
                    raise RuntimeError(
                        f"Spatial split failed for class {c}: "
                        f"dev={len(dev)}, test={len(te)}. "
                        f"Try larger dev block ratio, coarser grid, or reduce n_train/n_val."
                    )

                train_part = dev[:self.n_train_per_class]
                val_part = dev[self.n_train_per_class:required_dev]
                test_part = te

            else:
                if len(tr) < self.n_train_per_class or len(va) < self.n_val_per_class or len(te) < 1:
                    raise RuntimeError(
                        f"Spatial split failed for class {c}: "
                        f"train={len(tr)}, val={len(va)}, test={len(te)}. "
                        f"Try spatial_grid=(6,6), reduce n_train/n_val, or disable buffer."
                    )

                train_part = tr[:self.n_train_per_class]
                val_part = va[:self.n_val_per_class]
                test_part = te

            self.train_indices.extend(train_part)
            self.val_indices.extend(val_part)
            self.test_indices.extend(test_part)

            self.class_counts[c] = {
                "train_pool": len(tr),
                "val_pool": len(va),
                "test_pool": len(te),
                "train": len(train_part),
                "val": len(val_part),
                "test": len(test_part),
            }

    @staticmethod
    def _make_block_id_map(h: int, w: int, gh: int, gw: int):
        block_id_map = np.zeros((h, w), dtype=np.int64)
        block_coords = {}
        bid = 0
        row_edges = np.linspace(0, h, gh + 1, dtype=int)
        col_edges = np.linspace(0, w, gw + 1, dtype=int)

        for br in range(gh):
            for bc in range(gw):
                r0, r1 = row_edges[br], row_edges[br + 1]
                c0, c1 = col_edges[bc], col_edges[bc + 1]
                block_id_map[r0:r1, c0:c1] = bid
                block_coords[bid] = (br, bc)
                bid += 1
        return block_id_map, block_coords

    @staticmethod
    def _collect_split_pools(labels, block_id_map, train_blocks, val_blocks, test_blocks):
        train_pool, val_pool, test_pool = {}, {}, {}
        rows, cols = np.where(labels > 0)

        for i, j in zip(rows, cols):
            label = int(labels[i, j])
            bid = int(block_id_map[i, j])
            item = (int(i), int(j))

            if bid in train_blocks:
                train_pool.setdefault(label, []).append(item)
            elif bid in val_blocks:
                val_pool.setdefault(label, []).append(item)
            elif bid in test_blocks:
                test_pool.setdefault(label, []).append(item)

        return train_pool, val_pool, test_pool

    @staticmethod
    def _apply_center_buffer_fast(train_pool, val_pool, test_pool, radius: int, shape=None):
        """
        Fast spatial buffer using distance transform.
        Complexity: O(H*W), much faster than pairwise O(N^2).
        """
        if radius <= 0:
            return train_pool, val_pool, test_pool

        if shape is None:
            all_pts = []
            for pool in [train_pool, val_pool, test_pool]:
                for pts in pool.values():
                    all_pts.extend(pts)
            h = max(p[0] for p in all_pts) + 1
            w = max(p[1] for p in all_pts) + 1
        else:
            h, w = shape

        def pool_to_mask(pool):
            mask = np.zeros((h, w), dtype=bool)
            for pts in pool.values():
                for i, j in pts:
                    mask[i, j] = True
            return mask

        train_mask = pool_to_mask(train_pool)
        val_mask = pool_to_mask(val_pool)
        test_mask = pool_to_mask(test_pool)

        # distance to nearest pixel from other splits
        dist_to_val_test = distance_transform_edt(~(val_mask | test_mask))
        dist_to_train_test = distance_transform_edt(~(train_mask | test_mask))
        dist_to_train_val = distance_transform_edt(~(train_mask | val_mask))

        def filter_pool(pool, dist_map):
            out = {}
            for c, pts in pool.items():
                keep = []
                for i, j in pts:
                    if dist_map[i, j] > radius:
                        keep.append((i, j))
                out[c] = keep
            return out

        train_pool_f = filter_pool(train_pool, dist_to_val_test)
        val_pool_f = filter_pool(val_pool, dist_to_train_test)
        test_pool_f = filter_pool(test_pool, dist_to_train_val)

        return train_pool_f, val_pool_f, test_pool_f

    def _save_split_report(self, path: Path):
        report = {
            "dataset": self.dataset_name,
            "seed": self.seed,
            "patch_size": self.patch_size,
            "spatial_grid": list(self.spatial_grid),
            "train_block_ratio": self.train_block_ratio,
            "val_block_ratio": self.val_block_ratio,
            "use_spatial_buffer": self.use_spatial_buffer,
            "val_from_train_blocks": self.val_from_train_blocks,
            "n_train_per_class": self.n_train_per_class,
            "n_val_per_class": self.n_val_per_class,
            "num_train_samples": len(self.train_indices),
            "num_val_samples": len(self.val_indices),
            "num_test_samples": len(self.test_indices),
            "class_counts": self.class_counts,
            "protocol": "spatially_disjoint_test_closed_set",
            "note": (
                "Train/val/test samples are drawn from non-overlapping spatial blocks. "
                "When use_spatial_buffer=True, centers near split boundaries are removed "
                "to reduce patch overlap leakage."
            ),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    # ------------------------------------------------------------------
    # Dataset metadata and item access
    # ------------------------------------------------------------------
    def get_dataset_info(self) -> Dict:
        lidar_shape = self.data["lidar"].shape
        return {
            "hsi_channels": self.data["hsi"].shape[2],
            "lidar_channels": 1 if len(lidar_shape) == 2 else lidar_shape[2],
            "num_classes": len(np.unique(self.data["labels"])) - 1,
            "spatial_size": self.data["hsi"].shape[:2],
            "num_train_samples": len(self.train_indices),
            "num_val_samples": len(self.val_indices),
            "num_test_samples": len(self.test_indices),
            "n_train_per_class": self.n_train_per_class,
            "n_val_per_class": self.n_val_per_class,
            "seed": self.seed,
            "class_counts": self.class_counts,
            "split_protocol": "strict_spatially_disjoint_closed_set",
            "spatial_grid": self.spatial_grid,
            "use_spatial_buffer": self.use_spatial_buffer,
            "val_from_train_blocks": self.val_from_train_blocks,
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        h, w = self.data["labels"].shape
        half_patch = self.patch_size // 2

        i_min, i_max = i - half_patch, i + half_patch + 1
        j_min, j_max = j - half_patch, j + half_patch + 1

        hsi_channels = self.data["hsi"].shape[2]
        lidar_channels = 1 if len(self.data["lidar"].shape) == 2 else self.data["lidar"].shape[2]

        hsi_patch = np.zeros((self.patch_size, self.patch_size, hsi_channels), dtype=np.float32)
        lidar_patch = np.zeros((self.patch_size, self.patch_size, lidar_channels), dtype=np.float32)

        valid_i_min, valid_i_max = max(0, i_min), min(h, i_max)
        valid_j_min, valid_j_max = max(0, j_min), min(w, j_max)

        dest_i_min = valid_i_min - i_min
        dest_i_max = dest_i_min + (valid_i_max - valid_i_min)
        dest_j_min = valid_j_min - j_min
        dest_j_max = dest_j_min + (valid_j_max - valid_j_min)

        hsi_patch[dest_i_min:dest_i_max, dest_j_min:dest_j_max] = self.data["hsi"][
            valid_i_min:valid_i_max, valid_j_min:valid_j_max
        ]

        if len(self.data["lidar"].shape) == 2:
            lidar_patch[dest_i_min:dest_i_max, dest_j_min:dest_j_max, 0] = self.data["lidar"][
                valid_i_min:valid_i_max, valid_j_min:valid_j_max
            ]
        else:
            lidar_patch[dest_i_min:dest_i_max, dest_j_min:dest_j_max] = self.data["lidar"][
                valid_i_min:valid_i_max, valid_j_min:valid_j_max
            ]

        label = int(self.data["labels"][i, j]) - 1
        hsi_tensor = torch.from_numpy(hsi_patch.transpose(2, 0, 1))
        lidar_tensor = torch.from_numpy(lidar_patch.transpose(2, 0, 1))

        return {
            "hsi": hsi_tensor,
            "lidar": lidar_tensor,
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_dataset(
    dataset_name: str,
    batch_size: int = 32,
    patch_size: int = 7,
    num_workers: int = 4,
    data_root: Optional[str] = None,
    n_train_per_class: int = 7,
    n_val_per_class: int = 7,
    seed: int = 42,
    spatial_grid: Tuple[int, int] = (4, 4),
    train_block_ratio: float = 0.60,
    val_block_ratio: float = 0.00,
    max_split_tries: int = 50,
    use_spatial_buffer: bool = True,
    val_from_train_blocks: bool = True,
    split_cache_dir: Optional[str] = None,
    force_rebuild_split: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    train_dataset = HyperspectralDataset(
        dataset_name=dataset_name,
        split="train",
        patch_size=patch_size,
        data_root=data_root,
        n_train_per_class=n_train_per_class,
        n_val_per_class=n_val_per_class,
        seed=seed,
        spatial_grid=spatial_grid,
        train_block_ratio=train_block_ratio,
        val_block_ratio=val_block_ratio,
        max_split_tries=max_split_tries,
        use_spatial_buffer=use_spatial_buffer,
        val_from_train_blocks=val_from_train_blocks,
        split_cache_dir=split_cache_dir,
        force_rebuild_split=force_rebuild_split,
    )
    val_dataset = HyperspectralDataset(
        dataset_name=dataset_name,
        split="val",
        patch_size=patch_size,
        data_root=data_root,
        n_train_per_class=n_train_per_class,
        n_val_per_class=n_val_per_class,
        seed=seed,
        spatial_grid=spatial_grid,
        train_block_ratio=train_block_ratio,
        val_block_ratio=val_block_ratio,
        max_split_tries=max_split_tries,
        use_spatial_buffer=use_spatial_buffer,
        val_from_train_blocks=val_from_train_blocks,
        split_cache_dir=split_cache_dir,
        force_rebuild_split=False,
    )
    test_dataset = HyperspectralDataset(
        dataset_name=dataset_name,
        split="test",
        patch_size=patch_size,
        data_root=data_root,
        n_train_per_class=n_train_per_class,
        n_val_per_class=n_val_per_class,
        seed=seed,
        spatial_grid=spatial_grid,
        train_block_ratio=train_block_ratio,
        val_block_ratio=val_block_ratio,
        max_split_tries=max_split_tries,
        use_spatial_buffer=use_spatial_buffer,
        val_from_train_blocks=val_from_train_blocks,
        split_cache_dir=split_cache_dir,
        force_rebuild_split=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    info = train_dataset.get_dataset_info()
    return train_loader, val_loader, test_loader, info

def get_all_datasets_config():
    return {
        "houston2013": {
            "batch": 32,
            "patch": 7,
            "data_root": "D:/datasets/Houston, Trento, Muufl",
            "device": "cuda",
        },
        "muufl": {
            "batch": 32,
            "patch": 7,
            "data_root": "D:/datasets/Houston, Trento, Muufl",
            "device": "cuda",
        },
        "augsburg": {
            "batch": 32,
            "patch": 7,
            "data_root": "D:/datasets/Houston, Trento, Muufl",
            "device": "cuda",
        },
    }

if __name__ == "__main__":
    # Quick smoke test example:
    # python dataset_loader_spatial_disjoint.py
    for ds in ["houston2013", "muufl", "augsburg"]:
        try:
            tr, va, te, info = load_dataset(
                ds,
                batch_size=32,
                patch_size=7,
                num_workers=0,
                n_train_per_class=7,
                n_val_per_class=7,
                seed=42,
                spatial_grid=(4, 4),
                use_spatial_buffer=False,
            )
            print("\n", "=" * 80)
            print(ds)
            print(info)
            b = next(iter(tr))
            print("batch hsi:", b["hsi"].shape, "lidar:", b["lidar"].shape, "label:", b["label"].shape)
        except Exception as e:
            print(f"[WARN] {ds}: {e}")
