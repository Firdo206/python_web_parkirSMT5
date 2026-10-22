import shutil
import cv2
from app.config import DATASET_DIR

def save_samples(member_id: int, samples) -> int:
    """Timpa sampel lama member (daftar ulang), tulis sampel baru."""
    folder = DATASET_DIR / str(member_id)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(samples):
        cv2.imwrite(str(folder / f"{i:03d}.png"), img)
    return len(samples)

def delete_member(member_id: int) -> None:
    shutil.rmtree(DATASET_DIR / str(member_id), ignore_errors=True)