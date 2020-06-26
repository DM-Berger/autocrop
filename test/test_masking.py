import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from autocrop.constants import CC359, IMG1, IMG2, IMG3
from autocrop.visualize import make_masked


def test_mask(capsys) -> None:
    N = 4
    N_SAMPLES = 10
    np.random.seed(42)
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
