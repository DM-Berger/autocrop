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

from constants import TEST, TEST_MASK
from eigenimage import full_eigensignal


@unique
class Src(Enum):
    IMG = "IMG"
    RAW = "RAW"
    RAW_MEAN = "RAW_MEAN"


def make_cheaty_nii(orig: nib.Nifti1Image, array: np.array) -> nib.Nifti1Image:
    """clone the header and extraneous info from `orig` and data in `array`
    into a new Nifti1Image object, for plotting
    """
    # clone = deepcopy(orig)
    affine = orig.affine
    header = orig.header
    extra = orig.extra
    # return new_img_like(orig, array, copy_header=True)
    return nib.Nifti1Image(dataobj=array, affine=affine, header=header, extra=extra)


def slice_label(i: int, mids: ndarray, slicekey: str) -> str:
    quarts = mids // 2  # slices at first quarter of the way through
    quarts3_4 = mids + quarts  # slices 3/4 of the way through
    keymap = {"1/4": quarts, "1/2": mids, "3/4": quarts3_4}
    idx = keymap[slicekey]
    if i == 0:
        return f"[{idx[i]},:,:]"
    if i == 1:
        return f"[:,{idx[i]},:]"
    if i == 2:
        return f"[:,:,{idx[i]}]"

    f"[{idx[i]},:,:]", f"[:,{idx[i]},:]", f"[:,:,{idx[i]}]"
    raise IndexError("Only three dimensions supported.")


class BrainSlices:
    """Summary

    Parameters
    ----------
    img: eigenimage
    raw: original image
    """

    def __init__(
        self, img: ndarray, raw: ndarray, n_slices: int = 4, invert: bool = False, subtract_raw_mean: bool = False
    ):
        def get_slice_tuples(src: ndarray) -> Any:
            tuples = []
            for idx, name in zip(slicers, self.slice_positions):
                tuples.append((name, (src[idx[0], :, :, :], src[:, idx[1], :, :], src[:, :, idx[2], :])))
            return tuples

        self.orig = img
        self.img = img
        self.raw = raw

        self.subtract_raw_mean = subtract_raw_mean
        if subtract_raw_mean:
            zeros = np.sum(raw, axis=3) == 0.0
            mean_signal = np.mean(raw[~zeros], axis=0)
            raw_mean = np.copy(raw) - mean_signal
            raw_mean[zeros] = 0
            self.raw_mean = raw_mean

        if invert:
            inverted = img.max() - img
            inverted[inverted == inverted.max()] = 0
            self.img = inverted

        slice_base = np.array(self.img.shape[:-1]) // n_slices
        slicers = [slice_base * i for i in range(1, n_slices)]
        self.slice_positions = [f"{i}/{n_slices}" for i in range(1, n_slices)]
        self.imgs = OrderedDict(get_slice_tuples(self.img))
        self.targets = OrderedDict(get_slice_tuples(self.raw))

    def plot(self, t: int, vmin: int, ret: bool = False) -> Tuple[Figure, Axes]:
        nrows, ncols = 2, 1  # one row for each slice position
        all_eigs, all_raws = [], []
        for i in range(3):  # We want this first so middle images are middle
            for j, position in enumerate(self.slice_positions):
                img, target = self.imgs[position][i][:, :, t], self.targets[position][i][:, :, t]
                all_eigs.append(img)
                all_raws.append(target)
        fig: Figure
        axes: Axes
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)
        eig_img = np.concatenate(all_eigs, axis=1)
        raw_img = np.concatenate(all_raws, axis=1)

        # Consistently apply colormap since images are standardized but still
        # vary considerably in maximum and minimum values
        vals = eig_img[eig_img > 0]
        print("Eigenimage info for nonzero pixel values:")
        print(f"Max:  {np.max(vals)}")
        print(f"Min:  {np.min(vals)}")
        print(f"Mean: {np.mean(vals)}")
        print(f"Std.: {np.std(vals, ddof=1)}")

        eig_args = dict(vmin=np.percentile(vals, [vmin]), vmax=np.max(vals), cmap="gray")
        raw_args = dict(vmin=0.0, vmax=None, cmap="gray")

        axes[0].imshow(eig_img, **eig_args)
        axes[0].set_title("Eigenimage")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].imshow(raw_img, **raw_args)
        axes[1].set_title("Raw image")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        fig.tight_layout(h_pad=0)
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        if ret:
            return fig, axes
        plt.show()
        return fig, axes

    def animate_time(self, ts: List[int], vmin: int, outfile: Path = None) -> Tuple[Figure, Axes]:
        def get_slice_images(t: int) -> Tuple[ndarray, ndarray]:
            """Returns eig_img, raw_img"""
            all_eigs, all_raws = [], []
            for i in range(3):  # We want this first so middle images are middle
                for j, position in enumerate(self.slice_positions):
                    img, target = self.imgs[position][i][:, :, t], self.targets[position][i][:, :, t]
                    all_eigs.append(img)
                    all_raws.append(target)
            eig_img = np.concatenate(all_eigs, axis=1)
            raw_img = np.concatenate(all_raws, axis=1)
            return eig_img, raw_img

        nrows, ncols = 2, 1  # one row for each slice position
        fig: Figure
        axes: Axes
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)
        axes[0].set_title("Eigenimage")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_title("Raw image")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        fig.tight_layout(h_pad=0)
        fig.set_size_inches(w=12, h=4)
        fig.subplots_adjust(hspace=0.0, wspace=0.0)

        eig_img, raw_img = get_slice_images(ts[0])

        ims = []
        for t in ts:
            eig_img, raw_img = get_slice_images(t)

            vals = eig_img[eig_img > 0]
            eig_args = dict(vmin=np.percentile(vals, [vmin]), vmax=np.max(vals), cmap="gray")
            raw_args = dict(vmin=0.0, vmax=None, cmap="gray")

            # axes[0].set_title(f"Eigenimage (t = {t})")
            eig = axes[0].imshow(eig_img, **eig_args, animated=True)
            # axes[1].set_title(f"Raw image (t = {t})")
            raw = axes[1].imshow(raw_img, **raw_args, animated=True)
            ims.append([eig, raw])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100 if outfile is None else None)
        if outfile is None:
            plt.show()
        else:
            ani.save(outfile, codec="h264", dpi=256)

    def animate_space(
        self,
        ts: List[int],
        average: bool = False,
        invert: bool = False,
        vmin: int = None,
        vmax: int = None,
        cmap: str = "gray",
        interpolation: str = None,
        dpi: int = 256,
        fig_title: str = None,
        eig_title: str = None,
        outfile: Path = None,
    ) -> None:
        def get_slice(src: ndarray, t: int, ratio: float) -> ndarray:
            """Returns eig_img, raw_img"""
            if ratio < 0 or ratio > 1:
                raise ValueError("Invalid slice position")
            x_max, y_max, z_max = np.array(src.shape[:-1], dtype=int)
            x, y, z = np.array(np.floor(np.array(src.shape[:-1]) * ratio), dtype=int)
            x = int(10 + ratio * (x_max - 20))  # make x go from 10:-10 of x_max
            y = int(10 + ratio * (y_max - 20))  # make x go from 10:-10 of x_max
            x = x - 1 if x == x_max else x
            y = y - 1 if y == y_max else y
            z = z - 1 if z == z_max else z
            return np.concatenate([src[x, :, :, t], src[:, y, :, t], src[:, :, z, t]], axis=1)

        def get_vranges(image: ndarray, imgsrc: Src) -> Tuple[Optional[float], Optional[float]]:
            if imgsrc is Src.IMG:
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
            if imgsrc is Src.RAW:
                return np.min(image[image > 0]), None
            if imgsrc is Src.RAW_MEAN:
                return np.percentile(image, 5), None

        def init_frame(
            t: Optional[int], ratio: float, fig: Figure, ax: Axes, imgsrc: Src = Src.IMG
        ) -> Tuple[AxesImage, Colorbar, Text]:
            t = -1 if t is None else t
            if imgsrc is Src.IMG:
                image = get_slice(self.img, t, ratio)
                title = "Eigenimage"
            elif imgsrc is Src.RAW:
                image = get_slice(self.raw, t, ratio)
                title = f"Raw fMRI"
            elif imgsrc is Src.RAW_MEAN:
                image = get_slice(self.raw_mean, t, ratio)
                title = "Raw fMRI (minus mean signal)"

            vn, vm = get_vranges(image, imgsrc)
            eig_args = dict(vmin=vn, vmax=vm, cmap=cmap, interpolation=interpolation)
            im = ax.imshow(image, **eig_args, animated=True)
            ax.set_xticks([])
            ax.set_yticks([])
            title = ax.set_title(title)
            cb = fig.colorbar(im, ax=ax)
            return im, cb, title

        def update_axis(t: Optional[int], ratio: float, ax: Axes, im: AxesImage, imgsrc: Src = Src.IMG) -> None:
            t = -1 if t is None else t  # always use last frame
            if imgsrc is Src.IMG:
                image = get_slice(self.img, t, ratio)
            elif imgsrc is Src.RAW:
                image = get_slice(self.raw, t, ratio)
            elif imgsrc is Src.RAW_MEAN:
                image = get_slice(self.raw_mean, t, ratio)
            vn, vm = get_vranges(image, imgsrc)
            im.set_data(image)
            im.set_clim(vn, vm)
            # we don't have to update cb, it is linked

        # owe a lot to below for animating the colorbars
        # https://stackoverflow.com/questions/39472017/how-to-animate-the-colorbar-in-matplotlib
        def init(
            ts: List[int], title: Optional[str] = fig_title
        ) -> Tuple[Figure, Axes, List[AxesImage], List[Colorbar], List[Text]]:
            # one row for each time position, plus one for raw, one for raw mean
            # subtracted
            extra = int(self.subtract_raw_mean)
            nrows = len(ts) + 1 + extra
            ncols = 1
            fig: Figure
            axes: Axes
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)

            ims: List[AxesImage] = []
            cbs: List[Colorbar] = []
            titles: List[Text] = []

            for i, t in enumerate(ts):  # eigen images
                ax = axes.flat[i]
                im, cb, title = init_frame(t=t, ratio=0.0, fig=fig, ax=ax)
                ims.append(im)
                cbs.append(cb)
                titles.append(title)

            if self.subtract_raw_mean:
                im, cb, title = init_frame(t=None, ratio=0.0, fig=fig, ax=axes[-2], imgsrc=Src.RAW_MEAN)
                ims.append(im)
                cbs.append(cb)
                titles.append(title)
            im, cb, title = init_frame(t=None, ratio=0.0, fig=fig, ax=axes[-1], imgsrc=Src.RAW)
            ims.append(im)
            cbs.append(cb)
            titles.append(title)

            fig.tight_layout(h_pad=0)
            h = np.max([2 * (len(ts) + extra), 7])  # 2 inches per row, min 7 inches
            if fig_title is not None:
                fig.suptitle(fig_title)
            fig.set_size_inches(w=8, h=h)
            fig.subplots_adjust(hspace=0.2, wspace=0.0)
            return fig, axes, ims, cbs, titles

        N_FRAMES = 300
        ratios = np.linspace(0, 1, num=N_FRAMES)

        fig, axes, ims, cbs, titles = init(ts=ts)
        ani = None
        # awkward, but we need this defined after to close over the above variables
        def animate(f: int) -> None:
            ratio = ratios[f]
            for i, t in enumerate(ts):
                update_axis(t=t, ratio=ratio, ax=axes[i], im=ims[i])
            if self.subtract_raw_mean:
                update_axis(t=None, ratio=ratio, ax=axes[-2], im=ims[-2], imgsrc=Src.RAW_MEAN)
            update_axis(t=None, ratio=ratio, ax=axes[-1], im=ims[-1], imgsrc=Src.RAW)

        ani = animation.FuncAnimation(
            fig, animate, frames=N_FRAMES, blit=False, interval=12, repeat_delay=100 if outfile is None else None
        )

        if outfile is None:
            plt.show()
        else:

            def prog_logger(current_frame: int, total_frames: int = N_FRAMES) -> Any:
                if (current_frame % (total_frames // 10)) == 0 and (current_frame != 0):
                    print("Saving... {:2.1f}%".format(100 * current_frame / total_frames))

            ani.save(outfile, codec="h264", dpi=dpi, progress_callback=prog_logger)


def setup(log: bool = False) -> List[ndarray]:
    NPY_OUT = Path(__file__).parent.resolve() / "test_eig_img.npy"

    nii = nib.load(str(TEST))
    raw = nib.as_closest_canonical(nii)
    raw = nib.load(str(TEST)).get_fdata()

    np_img = np.clip(np.load(NPY_OUT), a_min=0, a_max=None)
    temp = make_cheaty_nii(nii, np_img)
    temp = nib.as_closest_canonical(temp)
    img = temp.get_fdata()

    full = full_eigensignal(raw, nib.load(str(TEST_MASK)).get_fdata(), covariance=True)
    delta = img / full  # keeps img positive
    nonzeros = delta[delta > 0]

    # mask = img == 0
    diffs = np.sum((full - img), axis=3, keepdims=True)
    diffs = diffs.max() - diffs
    # mask = diffs == diffs.max()
    # diffs[mask] = 0
    diffs_nonzeros = diffs[diffs > 0]
    np.clip(diffs, a_min=0, a_max=None, out=diffs)

    summed_delta = np.sum(delta, axis=3, keepdims=True)
    summed_delta_nonzeros = summed_delta[summed_delta > 0]
    summed = np.sum(img, axis=3, keepdims=True)
    summed_nonzeros = summed[summed > 0]

    if log:
        print("=" * 80)
        print("raw.mean()", raw.mean())
        print("raw.std()", raw.std(ddof=1))
        print("raw.max()", raw.max())
        print("=" * 80)
        print("delta.mean(): ", delta.mean())
        print("delta.std():  ", delta.std())
        print("delta.max():  ", delta.max())
        print("-" * 80)
        print("delta_nonzeros.mean(): ", nonzeros.mean())
        print("delta_nonzeros.std():  ", nonzeros.std())
        print("delta_nonzeros.min():  ", nonzeros.min())
        print("=" * 80)
        print("diffs.mean():   ", diffs.mean())
        print("diffs.std():    ", diffs.std())
        print("diffs negative: ", np.sum(diffs < 0))  # 51 voxels, who cares, rounding error
        print("diffs.min():    ", diffs.min())
        print("diffs.max():    ", diffs.max())
        print("-" * 80)
        print("diffs_nonzeros.mean(): ", diffs_nonzeros.mean())
        print("diffs_nonzeros.std():  ", diffs_nonzeros.std())
        print("diffs_nonzeros.min():  ", diffs_nonzeros.min())
        print("=" * 80)
        print("summed_delta.mean(): ", summed_delta.mean())
        print("summed_delta.std():  ", summed_delta.std())
        print("summed_delta.max():  ", summed_delta.max())
        print("-" * 80)
        print("summed_delta_nonzeros.mean(): ", summed_delta_nonzeros.mean())
        print("summed_delta_nonzeros.std():  ", summed_delta_nonzeros.std())
        print("summed_delta_nonzeros.min():  ", summed_delta_nonzeros.min())
        print("=" * 80)
        print("summed.mean(): ", summed.mean())
        print("summed.std():  ", summed.std())
        print("summed.max():  ", summed.max())
        print("-" * 80)
        print("summed_nonzeros.mean(): ", summed_nonzeros.mean())
        print("summed_nonzeros.std():  ", summed_nonzeros.std())
        print("summed_nonzeros.min():  ", summed_nonzeros.min())
        # print("Percentile | Value")
        # ps = np.arange(10, 100, 10)
        # vals = np.percentile(delta, ps)
        # for p, val in zip(ps, vals):
        # print("{:10.0f} | {:5.12f}".format(p, val))
        # delta = np.log(delta)
    return [
        raw,
        img,
        full,
        delta,
        nonzeros,
        diffs,
        diffs_nonzeros,
        summed_delta,
        summed_delta_nonzeros,
        summed,
        summed_nonzeros,
    ]


if __name__ == "__main__":
    [
        raw,
        img,
        full,
        delta,
        nonzeros,
        diffs,
        diffs_nonzeros,
        summed_delta,
        summed_delta_nonzeros,
        summed,
        summed_nonzeros,
    ] = setup()

    Ns = [18, 36, 54, 72]  # 148 // 2, timeseries is 148 points
    # Ns = [130, 112, 94, 72]  # 148 // 2, timeseries is 148 points
    for N in Ns:
        # deltaN = delta[:, :, :, N:]
        sumN = np.mean(img[:, :, :, :N], axis=3, keepdims=True)
        # nonzerosN = sumN[sumN > 0]
        mn = sumN.min()
        mx = sumN.max()
        closezerosN = sumN[np.isclose(sumN, mn)]
        nonzerosN = sumN[sumN > mn]
        onesN = sumN[np.isclose(sumN, mx)]
        vmin, vmax = np.percentile(nonzerosN, [1, 99])
        zeros = len(nonzeros.flat)
        print("=" * 80)
        print(f"Plotting `delta[:, :, :, {N}:]`")
        print(f"N Zeros by isclose:              {100*np.sum(np.isclose(nonzerosN, mn))/zeros}")
        print(f"N Zeros by isclose (atol=1e-12): {100*np.sum(np.isclose(nonzerosN, mn, atol=1e-12))/zeros}")
        print(f"N Zeros by isclose (atol=1e-10): {100*np.sum(np.isclose(nonzerosN, mn, atol=1e-10))/zeros}")
        print(f"N Zeros by isclose (atol=1e-8):  {100*np.sum(np.isclose(nonzerosN, mn, atol=1e-8))/zeros}")
        print(f"N Zeros by isclose (atol=1e-6):  {100*np.sum(np.isclose(nonzerosN, mn, atol=1e-6))/zeros}")
        print(f"% Ones by isclose:               {100*np.sum(np.isclose(sumN, mx))/zeros}")
        print(f"% Ones by isclose (atol=1e-12):  {100*np.sum(np.isclose(sumN, mx, atol=1e-12))/zeros}")
        print(f"% Ones by isclose (atol=1e-10):  {100*np.sum(np.isclose(sumN, mx, atol=1e-10))/zeros}")
        print(f"% Ones by isclose (atol=1e-8):   {100*np.sum(np.isclose(sumN, mx, atol=1e-8))/zeros}")
        print(f"% Ones by isclose (atol=1e-6):   {100*np.sum(np.isclose(sumN, mx, atol=1e-6))/zeros}")
        print(f"% Ones by isclose (atol=1e-4):   {100*np.sum(np.isclose(sumN, mx, atol=1e-4))/zeros}")
        print(f"  vrange: {vmax - vmin}")
        print(f"  sd:     {np.std(nonzerosN)}")
        print(f"  min:    {nonzerosN.min()}")
        # sbn.distplot(nonzerosN.flat, kde=False, bins=np.linspace(nonzerosN.min(), nonzerosN.max(), 1000))
        # plt.xlim(vmin, nonzerosN.max())
        # if N != 72:
        #     plt.show(block=False)
        # else:
        #     plt.show()
        continue

        slices = BrainSlices(sumN, raw, n_slices=1, subtract_raw_mean=True)
        slices.animate_space(
            [0],
            average=False,
            invert=False,
            vmin=vmin,
            vmax=vmax,
            dpi=100,
            cmap="gray",
            interpolation="hamming",
            # fig_title=f"Σ δ of {148-N} Largest Eigenvalues",
            fig_title=f"Σ of {N} Smallest Eigenvalues",
            outfile=Path(f"eigen_sum_smallest{N}.mp4"),
            # outfile=Path(f"eigen_delta_ratio_largest{148-N}.mp4"),
        )

    sys.exit(0)

    Ns = [18, 36, 54, 72]  # 148 // 2, timeseries is 148 points
    # Ns = [130, 112, 94, 72]  # 148 // 2, timeseries is 148 points
    for N in Ns:
        # deltaN = delta[:, :, :, N:]
        deltaN = delta[:, :, :, :N]
        sumN = np.mean(deltaN, axis=3, keepdims=True)
        # nonzerosN = sumN[sumN > 0]
        closezerosN = sumN[np.isclose(sumN, 0)]
        nonzerosN = sumN[sumN > 0]
        onesN = sumN[np.isclose(sumN, 1.0)]
        vmin, vmax = np.percentile(nonzerosN, [1, 99])
        zeros = len(nonzeros.flat)
        print("=" * 80)
        print(f"Plotting `delta[:, :, :, {N}:]`")
        print(f"N Zeros by isclose:              {100*np.sum(np.isclose(nonzerosN, 0))/zeros}")
        print(f"N Zeros by isclose (atol=1e-12): {100*np.sum(np.isclose(nonzerosN, 0, atol=1e-12))/zeros}")
        print(f"N Zeros by isclose (atol=1e-10): {100*np.sum(np.isclose(nonzerosN, 0, atol=1e-10))/zeros}")
        print(f"N Zeros by isclose (atol=1e-8):  {100*np.sum(np.isclose(nonzerosN, 0, atol=1e-8))/zeros}")
        print(f"N Zeros by isclose (atol=1e-6):  {100*np.sum(np.isclose(nonzerosN, 0, atol=1e-6))/zeros}")
        print(f"% Ones by isclose:               {100*np.sum(np.isclose(sumN, 1))/zeros}")
        print(f"% Ones by isclose (atol=1e-12):  {100*np.sum(np.isclose(sumN, 1, atol=1e-12))/zeros}")
        print(f"% Ones by isclose (atol=1e-10):  {100*np.sum(np.isclose(sumN, 1, atol=1e-10))/zeros}")
        print(f"% Ones by isclose (atol=1e-8):   {100*np.sum(np.isclose(sumN, 1, atol=1e-8))/zeros}")
        print(f"% Ones by isclose (atol=1e-6):   {100*np.sum(np.isclose(sumN, 1, atol=1e-6))/zeros}")
        print(f"% Ones by isclose (atol=1e-4):   {100*np.sum(np.isclose(sumN, 1, atol=1e-4))/zeros}")
        print(f"  vrange: {vmax - vmin}")
        print(f"  sd:     {np.std(nonzerosN)}")
        print(f"  min:    {nonzerosN.min()}")
        # sbn.distplot(nonzerosN.flat, kde=False, bins=np.linspace(nonzerosN.min(), nonzerosN.max(), 1000))
        # plt.xlim(vmin, nonzerosN.max())
        # if N != 72:
        #     plt.show(block=False)
        # else:
        #     plt.show()
        # continue

        slices = BrainSlices(sumN, raw, n_slices=1, subtract_raw_mean=True)
        slices.animate_space(
            [0],
            average=False,
            invert=False,
            vmin=vmin,
            vmax=vmax,
            dpi=100,
            cmap="gray",
            interpolation="hamming",
            # fig_title=f"Σ δ of {148-N} Largest Eigenvalues",
            fig_title=f"Σ δ of {N} Smallest Eigenvalues",
            outfile=Path(f"eigen_delta_ratio_smallest{N}.mp4"),
            # outfile=Path(f"eigen_delta_ratio_largest{148-N}.mp4"),
        )

    # slices = BrainSlices(diffs, raw, n_slices=1, subtract_raw_mean=True)
    # vmin, vmax = np.percentile(diffs_nonzeros.flat, [5, 95])
    # slices.animate_space(
    #     [0],
    #     average=False,
    #     invert=False,
    #     vmin=vmin,
    #     vmax=vmax,
    #     dpi=256,
    #     cmap="gray",
    #     interpolation="hamming",
    #     fig_title="Sum of deltas (Differences) Across all Eigenvalues",
    #     outfile=Path("eigen_delta_diff_summed_all.mp4"),
    # )

    # slices = BrainSlices(summed_delta, raw, n_slices=1, subtract_raw_mean=True)
    # vmin, vmax = np.percentile(summed_delta_nonzeros.flat, [5, 100])
    # slices.animate_space(
    #     [0],
    #     average=False,
    #     invert=False,
    #     vmin=vmin,
    #     vmax=vmax,
    #     dpi=256,
    #     cmap="gray",
    #     interpolation="hamming",
    #     fig_title="Sum of deltas (Ratio) Across all Eigenvalues",
    #     outfile=Path("eigen_delta_ratio_summed_all.mp4"),
    # )

    # slices = BrainSlices(summed, raw, n_slices=1, subtract_raw_mean=True)
    # vmin, vmax = np.percentile(summed_nonzeros, [5, 95])
    # slices.animate_space(
    #     [0],
    #     average=False,
    #     invert=False,
    #     vmin=5,
    #     vmax=95,
    #     dpi=256,
    #     cmap="gray",
    #     interpolation="hamming",
    #     fig_title="Sum of all raw Eigenvalues",
    #     outfile=Path("eigen_summed_all.mp4"),
    # )

    # slices = BrainSlices(diffs, raw, n_slices=1, subtract_raw_mean=True, invert=True)
    # imax = diffs.max()
    # vmin, vmax = np.percentile(imax - diffs_nonzeros, [5, 95])
    # slices.animate_space(
    #     [0],
    #     average=False,
    #     invert=False,
    #     vmin=vmin,
    #     vmax=vmax,
    #     dpi=256,
    #     cmap="gray",
    #     interpolation="hamming",
    #     fig_title="Inverted Sum of deltas (Differences)",
    #     outfile=Path("eigen_delta_diff_summed_all_inverted.mp4"),
    # )

    # slices = BrainSlices(summed_delta, raw, n_slices=1, subtract_raw_mean=True, invert=True)
    # imax = np.max(summed_delta)
    # vmin, vmax = np.percentile(imax - summed_delta_nonzeros, [5, 95])
    # slices.animate_space(
    #     [0],
    #     average=False,
    #     invert=False,
    #     vmin=vmin,
    #     vmax=vmax,
    #     dpi=256,
    #     cmap="gray",
    #     interpolation="hamming",
    #     fig_title="Inverted Sum of deltas (Ratio)",
    #     outfile=Path("eigen_delta_ratio_summed_all_inverted.mp4"),
    # )

    # slices = BrainSlices(summed, raw, n_slices=1, subtract_raw_mean=True, invert=True)
    # imax = summed.max()
    # vmin, vmax = np.percentile(imax - summed_nonzeros, [5, 95])
    # slices.animate_space(
    #     [0],
    #     average=False,
    #     invert=False,
    #     vmin=vmin,
    #     vmax=vmax,
    #     dpi=256,
    #     cmap="gray",
    #     interpolation="hamming",
    #     fig_title="Inverted Sum",
    #     outfile=Path("eigen_summed_all_inverted.mp4"),
    # )
