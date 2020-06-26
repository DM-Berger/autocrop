"""
NOTES
-----
The resulting images are highly bimodal at each time slices. That is,
regardless of whether looking at the delta image (diff from full eigensignal)
or just the deletion signals, the vast majority of values are zero, and the
remaining will lie almost entirely near the maximum value.
"""
import sys
from collections import OrderedDict
from enum import Enum, unique
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sbn
from matplotlib import animation
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from matplotlib.pyplot import Axes, Figure
from matplotlib.text import Text
from numpy import ndarray


@unique
class Src(Enum):
    IMG = "IMG"
    RAW = "RAW"
    RAW_MEAN = "RAW_MEAN"


def make_masked(img: ndarray, mask: ndarray, imin: Any = None, imax: Any = None) -> ndarray:
    img3d = np.zeros([*img.shape, 3], dtype=int)
    imin = img.min() if imin is None else imin
    imax = img.max() if imax is None else imax
    scaled = np.array(((img - imin) / (imax - imin)) * 255, dtype=int)
    if len(img.shape) == 3:
        img3d[:, :, :, 0] = scaled
        img3d[:, :, :, 1] = scaled
        masked = scaled + 200 * np.array(mask, dtype=int)
        img3d[:, :, :, 2] = masked
        return img3d
    # if len(img.shape) == 3:
    #     img3d[:, :, 0] = scaled
    #     img3d[:, :, 1] = scaled
    #     masked = scaled + 200 * np.array(mask, dtype=int)
    #     img3d[:, :, 2] = masked
    #     return img3d
    raise ValueError("Only accepts 1-channel or 3-channel images")


def pad_to_cube(img: ndarray) -> ndarray:
    """assumes images is 1-channel and 3D"""
    pad_shape = np.max(img.shape) * np.ones([3], dtype=int)
    full_padlengths = pad_shape - np.array(img.shape)
    pads = np.array([[padlength // 2, padlength // 2 + padlength % 2] for padlength in full_padlengths])
    return np.pad(img, pads)


class BrainSlices:
    """Summary

    Parameters
    ----------
    img: eigenimage
        If img is multichannel, assumes channels are last
    raw: original image
    """

    def __init__(
        self, img: ndarray, masks: List[ndarray], masknames: List[str], n_slices: int = 4, invert: bool = False
    ):
        def get_slice_tuples(src: ndarray) -> Any:
            tuples = []
            for idx, name in zip(slicers, self.slice_positions):
                tuples.append((name, (src[idx[0], :, :], src[:, idx[1], :], src[:, :, idx[2]])))
            return tuples

        self.orig = img
        self.img = img
        self.masks = masks
        self.masknames = masknames
        self.mask_args = dict(vmin=0.0, vmax=1.0, cmap="winter", alpha=0.5)

        if invert:
            inverted = img.max() - img
            inverted[inverted == inverted.max()] = 0
            self.img = inverted

        slice_base = np.array(self.img.shape) // n_slices
        slicers = [slice_base * i for i in range(1, n_slices)]
        self.slice_positions = [f"{i}/{n_slices}" for i in range(1, n_slices)]
        self.imgs = OrderedDict(get_slice_tuples(self.img))
        self.mask_slices = [OrderedDict(get_slice_tuples(m)) for m in self.masks]
        self.maskeds = [make_masked(self.img, mask) for mask in self.masks]

    def plot(self, vmin: int = 0, ret: bool = False) -> Tuple[Figure, Axes]:
        nrows, ncols = len(self.masks), 1  # one row for each slice position
        all_imgs, all_masks = [], [[] for _ in self.masks]
        for i in range(3):  # We want this first so middle images are middle
            for j, position in enumerate(self.slice_positions):
                img = self.imgs[position][i][:, :]
                all_imgs.append(img)
        for k, mask in enumerate(self.mask_slices):
            for i in range(3):  # We want this first so middle images are middle
                for j, position in enumerate(self.slice_positions):
                    mask_slice = mask[position][i][:, :]
                    all_masks[k].append(mask_slice)

        fig: Figure
        axes: Axes
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)
        slice_img = np.concatenate(all_imgs, axis=1)
        mask_imgs = [np.concatenate(mask_slices, axis=1) for mask_slices in all_masks]

        # Consistently apply colormap since images are standardized but still
        # vary considerably in maximum and minimum values
        vals = slice_img[slice_img > 0]
        print("Image info for nonzero pixel values:")
        print(f"Max:  {np.max(vals)}")
        print(f"Min:  {np.min(vals)}")
        print(f"Mean: {np.mean(vals)}")
        print(f"Std.: {np.std(vals, ddof=1)}")

        img_args = dict(vmin=np.percentile(vals, [vmin]), vmax=np.max(vals), cmap="gray")

        for ax, mask_img, maskname in zip(axes.flat, mask_imgs, self.masknames):
            ax.imshow(make_masked(slice_img, mask_img))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(maskname)

        fig.tight_layout(h_pad=0)
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        if ret:
            return fig, ax
        plt.show()
        return fig, ax

    def animate_masks(
        self,
        invert: bool = False,
        vmin: int = None,
        vmax: int = None,
        cmap: str = "gray",
        interpolation: str = None,
        dpi: int = 100,
        n_frames: int = 300,
        fig_title: str = None,
        outfile: Path = None,
    ) -> None:
        def get_slice(src: ndarray, ratio: float) -> ndarray:
            """Returns eig_img, raw_img"""
            if ratio < 0 or ratio > 1:
                raise ValueError("Invalid slice position")
            if len(src.shape) == 3:
                x_max, y_max, z_max = np.array(src.shape, dtype=int)
                x, y, z = np.array(np.floor(np.array(src.shape) * ratio), dtype=int)
            elif len(src.shape) == 4:
                x_max, y_max, z_max, _ = np.array(src.shape, dtype=int)
                x, y, z = np.array(np.floor(np.array(src.shape[:-1]) * ratio), dtype=int)
            x = int(10 + ratio * (x_max - 20))  # make x go from 10:-10 of x_max
            y = int(10 + ratio * (y_max - 20))  # make x go from 10:-10 of x_max
            x = x - 1 if x == x_max else x
            y = y - 1 if y == y_max else y
            z = z - 1 if z == z_max else z
            return np.concatenate([src[x, :, :], src[:, y, :], src[:, :, z]], axis=1)

        def get_vranges() -> Tuple[Optional[float], Optional[float]]:
            image = self.img
            vals = image[image > 0]
            if vmax is None:
                vm = None
            if isinstance(vmax, int):
                vm = np.percentile(vals, vmax)
            if isinstance(vmax, float):
                vm = vmax
            if vmin is None:
                vn = None
            if isinstance(vmin, int):
                vn = np.percentile(vals, vmin)
            if isinstance(vmin, float):
                vn = vmin
            return vn, vm

        def init_frame(src: ndarray, ratio: float, fig: Figure, ax: Axes) -> Tuple[AxesImage, Colorbar, Text]:
            # image = get_slice(self.img, ratio)
            # mask = get_slice(src_mask, ratio)
            image = get_slice(src, ratio)
            title = "Cropped Voxels"

            vn, vm = get_vranges()
            # plot_args = dict(vmin=vn, vmax=vm, cmap=cmap, interpolation=interpolation)
            # im = ax.imshow(image, **plot_args, animated=True)
            im = ax.imshow(image, animated=True)
            # mask = ax.imshow(mask, **self.mask_args, animated=True)
            ax.set_xticks([])
            ax.set_yticks([])
            title = ax.set_title(title)
            cb = fig.colorbar(im, ax=ax)
            # return im, mask, cb, title
            return im, cb, title

        def update_axis(src: ndarray, ratio: float, ax: Axes, im: AxesImage) -> None:
            image = get_slice(src, ratio)
            vn, vm = get_vranges()
            im.set_data(image)
            im.set_clim(vn, vm)
            # we don't have to update cb, it is linked

        # owe a lot to below for animating the colorbars
        # https://stackoverflow.com/questions/39472017/how-to-animate-the-colorbar-in-matplotlib
        def init() -> Tuple[Figure, Axes, List[AxesImage], List[Colorbar], List[Text]]:
            fig: Figure
            axes: Axes
            fig, axes = plt.subplots(nrows=len(self.maskeds), ncols=1, sharex=False, sharey=False)

            ims: List[AxesImage] = []
            cbs: List[Colorbar] = []
            titles: List[Text] = []

            for ax, masked, title in zip(axes.flat, self.maskeds, self.masknames):
                im, cb, title = init_frame(src=masked, ratio=0.0, fig=fig, ax=ax)
                ims.append(im)
                cbs.append(cb)
                titles.append(title)

            if fig_title is not None:
                fig.suptitle(fig_title)
            fig.tight_layout(h_pad=0)
            fig.set_size_inches(w=8, h=3)
            fig.subplots_adjust(hspace=0.2, wspace=0.0)
            return fig, axes, ims, cbs, titles

        N_FRAMES = n_frames
        ratios = np.linspace(0, 1, num=N_FRAMES)

        fig, axes, ims, cbs, titles = init()
        ani = None

        # awkward, but we need this defined after to close over the above variables
        def animate(f: int) -> None:
            ratio = ratios[f]
            for im, masked, ax in zip(ims, self.maskeds, axes):
                update_axis(src=masked, ratio=ratio, ax=ax, im=im)

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=N_FRAMES,
            blit=False,
            interval=3600 / N_FRAMES,
            repeat_delay=100 if outfile is None else None,
        )

        if outfile is None:
            plt.show()
        else:

            def prog_logger(current_frame: int, total_frames: int = N_FRAMES) -> Any:
                if (current_frame % (total_frames // 10)) == 0 and (current_frame != 0):
                    print("Saving... {:2.1f}%".format(100 * current_frame / total_frames))

            ani.save(outfile, codec="h264", dpi=dpi, progress_callback=prog_logger)
