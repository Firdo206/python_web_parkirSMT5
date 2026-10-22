import threading
import cv2
import numpy as np
from app.config import DATASET_DIR, LBPH_PARAMS, MODEL_PATH

_lock = threading.Lock()
_recognizer = None

def _load_dataset():
    images, labels = [], []
    if DATASET_DIR.exists():
        for folder in sorted(DATASET_DIR.iterdir()):
            if not (folder.is_dir() and folder.name.isdigit()):
                continue
            for f in sorted(folder.glob("*.png")):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(int(folder.name))
    return images, np.array(labels, dtype=np.int32)

def train() -> int:
    """Latih ulang model dari seluruh dataset. Return jumlah citra training."""
    global _recognizer
    with _lock:
        images, labels = _load_dataset()
        if not images:
            _recognizer = None
            MODEL_PATH.unlink(missing_ok=True)
            return 0

        rec = cv2.face.LBPHFaceRecognizer_create(**LBPH_PARAMS)
        rec.train(images, labels)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_name("lbph.tmp.yml")
        rec.write(str(tmp))
        tmp.replace(MODEL_PATH)          # ganti model secara atomik
        _recognizer = rec
        return len(images)

def load_model() -> bool:
    global _recognizer
    with _lock:
        if not MODEL_PATH.exists():
            return False
        rec = cv2.face.LBPHFaceRecognizer_create(**LBPH_PARAMS)
        rec.read(str(MODEL_PATH))
        _recognizer = rec
        return True

def predict(face):
    """Return (member_id, distance) atau None jika model belum ada."""
    with _lock:
        if _recognizer is None:
            return None
        label, distance = _recognizer.predict(face)
    return int(label), float(distance)