from pathlib import Path
from typing import List, Tuple

import nibabel as nib
from numpy import ndarray

from autocrop.visualize import BrainSlices

DATA_ROOT = Path(__file__).parent / "data"
IMG1 = DATA_ROOT / "sub-cntrl01_T1w.nii.gz"
IMG2 = DATA_ROOT / "sub-pddn01_T1w.nii.gz"


def setup() -> Tuple[ndarray, ndarray]:
    img1 = nib.load(str(IMG1)).get_fdata()
    img2 = nib.load(str(IMG2)).get_fdata()
    return img1, img2


def test_cube(log: bool = False) -> List[ndarray]:
    pass
