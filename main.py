from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import face_recognition
import easyocr
from ultralytics import YOLO
import uuid
import os

app = FastAPI(title="ParkVisi AI Service")

# Dimuat sekali saat service start, bukan setiap request
yolo_model = YOLO("yolov8n.pt")       # model umum, nanti bisa diganti model custom plat/wajah
ocr_reader = easyocr.Reader(['en'])

os.makedirs("temp", exist_ok=True)


def read_image_to_array(file_bytes: bytes):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "File bukan gambar yang valid")
    return img


@app.get("/health")
def health():
    return {"status": "ok"}


# ====== ENDPOINT 1: Dipakai saat ADMIN mendaftarkan wajah member ======
@app.post("/extract-embedding")
async def extract_embedding(photo: UploadFile = File(...)):
    content = await photo.read()
    img = read_image_to_array(content)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)

    if len(face_locations) == 0:
        return JSONResponse(status_code=422, content={"error": "Tidak ada wajah terdeteksi di foto"})

    if len(face_locations) > 1:
        return JSONResponse(status_code=422, content={"error": "Foto mengandung lebih dari satu wajah"})

    encodings = face_recognition.face_encodings(rgb, face_locations)
    embedding = encodings[0].tolist()  # 128 angka desimal

    return {"embedding": embedding, "face_count": 1}


# ====== ENDPOINT 2: Dipakai saat ESP8266 mengirim foto di gerbang ======
@app.post("/detect-access")
async def detect_access(photo: UploadFile = File(...)):
    content = await photo.read()
    img = read_image_to_array(content)

    # simpan sementara untuk referensi/log, opsional
    filename = f"temp/{uuid.uuid4().hex}.jpg"
    cv2.imwrite(filename, img)

    result = {
        "plate_text": None,
        "plate_confidence": None,
        "face_embedding": None,
        "face_detected": False,
    }

    # --- YOLO: cari area plat & wajah ---
    detections = yolo_model(img, verbose=False)[0]

    plate_crop = None
    for box in detections.boxes:
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # NOTE: model yolov8n.pt bawaan ini TIDAK dilatih khusus
        # untuk "plat nomor". Untuk hasil akurat, nanti perlu model
        # custom (dilatih sendiri dari dataset plat Indonesia) atau
        # pustaka deteksi plat siap pakai. Ini placeholder dulu.
        if label in ("car", "truck", "motorcycle"):
            plate_crop = img[y1:y2, x1:x2]

    # --- OCR plat (kalau ada crop) ---
    if plate_crop is not None and plate_crop.size > 0:
        ocr_result = ocr_reader.readtext(plate_crop)
        if ocr_result:
            best = max(ocr_result, key=lambda r: r[2])  # ambil confidence tertinggi
            result["plate_text"] = best[1].upper().replace(" ", "")
            result["plate_confidence"] = float(best[2])

    # --- Face recognition ---
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)

    if face_locations:
        encodings = face_recognition.face_encodings(rgb, face_locations)
        result["face_embedding"] = encodings[0].tolist()
        result["face_detected"] = True

    os.remove(filename)  # hapus file sementara

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)