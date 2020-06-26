from typing import Any

import hdbscan
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sklearn.cluster import DBSCAN, MiniBatchKMeans

from autocrop.constants import CC359, IMG1, IMG2, IMG3
from autocrop.visualize import BrainSlices, make_masked, pad_to_cube


def test_mask(capsys) -> None:
    N = 4
    N_SAMPLES = 10
    # np.random.seed(42)
    fig, axes = plt.subplots(ncols=N_SAMPLES, nrows=2, squeeze=False)
    for i, path in enumerate(np.random.choice(CC359, size=N_SAMPLES, replace=False)):
        full = nib.load(str(path)).get_fdata()
        flat = full.ravel()
        img = full[full.shape[0] // 2, :, :]
        km = MiniBatchKMeans(N, batch_size=1000).fit(flat.reshape(-1, 1))
        gs = [km.labels_ == i for i in range(N)]
        maxs = sorted([np.max(flat[g]) for g in gs])
        thresh = maxs[0]

        mask = np.zeros_like(img, dtype=int)
        mask[img > thresh] = 1
        masked = make_masked(img, mask)

        naive_mask = img > np.percentile(full, 80)
        naive_masked = make_masked(img, naive_mask)
        axes[0, i].imshow(masked)
        axes[1, i].imshow(naive_masked)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        with capsys.disabled():
            print(f"Finished image {i}")
    fig.suptitle(f"Top: K-Means ({N}) -- Bottom: > 200")
    fig.tight_layout(h_pad=0)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.show()


def test_mask_full(capsys: Any) -> None:
    N = 4
    N_SAMPLES = 3
    # np.random.seed(42)
    for i, path in enumerate(np.random.choice(CC359, size=N_SAMPLES, replace=False)):
        img = pad_to_cube(nib.load(str(path)).get_fdata())
        flat = img.ravel()
        km = MiniBatchKMeans(N, batch_size=1000).fit(flat.reshape(-1, 1))
        gs = [km.labels_ == i for i in range(N)]
        maxs = sorted([np.max(flat[g]) for g in gs])
        thresh = maxs[0]

        mask = np.zeros_like(img, dtype=int)
        mask[img > thresh] = 1
        # masked = make_masked(img, mask)

        naive_mask = np.zeros_like(img, dtype=int)
        naive_mask[img > np.percentile(img, 80)] = 1

        slices = BrainSlices(img, masks=[mask, naive_mask], masknames=["kmeans", "percentile80"], n_slices=7)
        slices.plot()
        # naive_masked = make_masked(img, naive_mask)
        with capsys.disabled():
            print(f"Finished image {i}")


def test_mask_video(capsys: Any) -> None:
    N = 4
    N_SAMPLES = 1
    np.random.seed(42)
    for i, path in enumerate(np.random.choice(CC359, size=N_SAMPLES, replace=False)):
        img = pad_to_cube(nib.load(str(path)).get_fdata())
        flat = img.ravel()
        km = MiniBatchKMeans(N, batch_size=1000).fit(flat.reshape(-1, 1))
        gs = [km.labels_ == i for i in range(N)]
        maxs = sorted([np.max(flat[g]) for g in gs])
        thresh = maxs[0]

        mask = np.zeros_like(img, dtype=int)
        mask[img > thresh] = 1
        # masked = make_masked(img, mask)

        naive_mask = np.zeros_like(img, dtype=int)
        naive_mask[img > np.percentile(img, 80)] = 1

        slices = BrainSlices(img, masks=[mask, naive_mask], masknames=["kmeans", "percentile80"], n_slices=7)
        # naive_masked = make_masked(img, naive_mask)
        with capsys.disabled():
            slices.animate_masks(dpi=150, n_frames=100, outfile=f"mask_compare{i}.mp4")
            print(f"Finished image {i}")


def test_dbscan(capsys) -> None:
    N = 4
    N_SAMPLES = 10
    np.random.seed(42)
    fig, axes = plt.subplots(ncols=N_SAMPLES, nrows=2, squeeze=False)
    for i, path in enumerate(np.random.choice(CC359, size=N_SAMPLES, replace=False)):
        full = nib.load(str(path)).get_fdata()
        flat = full.ravel()
        img = full[full.shape[0] // 2, :, :]
        dbs = DBSCAN(eps=0.5, n_jobs=-1).fit(full)
        gs = [dbs.labels_ == i for i in range(N)]
        maxs = sorted([np.max(flat[g]) for g in gs])
        thresh = maxs[0]

        mask = np.zeros_like(img, dtype=int)
        mask[img > thresh] = 1
        masked = make_masked(img, mask)

        naive_mask = img > 200
        naive_masked = make_masked(img, naive_mask)
        axes[0, i].imshow(masked)
        axes[1, i].imshow(naive_masked)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        with capsys.disabled():
            print(f"Finished image {i}")
    fig.suptitle(f"Top: K-Means ({N}) -- Bottom: > 200")
    fig.tight_layout(h_pad=0)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.show()


def test_hdbscan(capsys) -> None:
    N = 4
    N_SAMPLES = 5
    np.random.seed(42)
    fig, axes = plt.subplots(ncols=N_SAMPLES, nrows=2, squeeze=False)
    for i, path in enumerate(np.random.choice(CC359, size=N_SAMPLES, replace=False)):
        full = nib.load(str(path)).get_fdata()
        img = full[full.shape[0] // 2, :, :]
        hdb = hdbscan.HDBSCAN(min_cluster_size=200, core_dist_n_jobs=-1, allow_single_cluster=True)
        hdb.fit(full)
        gs = [hdb.labels_ == i for i in range(N)]
        maxs = sorted([np.max(full[g]) for g in gs])
        thresh = maxs[0]

        mask = np.zeros_like(img, dtype=int)
        mask[img > thresh] = 1
        masked = make_masked(img, mask)

        naive_mask = img > 200
        naive_masked = make_masked(img, naive_mask)
        axes[0, i].imshow(masked)
        axes[1, i].imshow(naive_masked)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        with capsys.disabled():
            print(f"Finished image {i}")
    fig.suptitle(f"Top: K-Means ({N}) -- Bottom: > 200")
    fig.tight_layout(h_pad=0)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.show()
