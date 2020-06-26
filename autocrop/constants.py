from pathlib import Path
from glob import glob

DATA_ROOT = Path(__file__).parent.parent / "data"
IMG1 = DATA_ROOT / "sub-cntrl01_T1w.nii.gz"
IMG2 = DATA_ROOT / "sub-pddn01_T1w.nii.gz"
IMG3 = DATA_ROOT / "CC0047_philips_15_54_F.nii.gz"

CC359 = [Path(p) for p in glob("/home/derek/Desktop/unet-segmentation/data/CC359/Reconstructed/Original/*.nii.gz")]
