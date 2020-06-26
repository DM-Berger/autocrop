from pathlib import Path

DATA_ROOT = Path("/home/derek/Desktop/eigencontribution/data/preproc").resolve()

PARK = DATA_ROOT / "Park+Controls"

TEST = PARK / "parkinsons/sub-RC4207_ses-1_task-ANT_run-3_fullpre_stripped.nii.gz"
TEST_MASK = PARK / "parkinsons/sub-RC4207_ses-1_task-ANT_run-3_fullpre_stripped_mask.nii.gz"

TEST_OUT = DATA_ROOT / "TEST_OUTPUTS"
