from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "faces"
MODEL_PATH = DATA_DIR / "models" / "lbph.yml"

FACE_SIZE = (200, 200)
MIN_FACE_PX = 80
LBPH_PARAMS = dict(radius=1, neighbors=8, grid_x=8, grid_y=8)
UNKNOWN_THRESHOLD = 70.0   # makin kecil makin mirip; kalibrasi dengan data nyata

# Pendaftaran: browser mengirim ~30 frame, tidak semuanya berisi wajah.
MIN_VALID_FRAMES = 15      # minimal frame yang wajahnya terdeteksi Haar cascade
# Dengan 30 frame asli, augmentasi tidak wajib. Tiap citra training memakan
# ~64 KB di model (grid 8x8), jadi augmentasi mempercepat membengkaknya model.
AUGMENT = False