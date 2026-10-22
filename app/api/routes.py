from flask import Blueprint, jsonify, request

from app.config import AUGMENT, MIN_VALID_FRAMES, UNKNOWN_THRESHOLD
from app.intelligence import dataset, lbph
from app.vision import detector, preprocess

bp = Blueprint("api", __name__)


def _prepare_face(file):
    """Citra mentah -> Haar cascade (cari wajah) -> crop -> normalisasi. Return (wajah, pesan_error)."""
    img = preprocess.decode_image(file.read())
    if img is None:
        return None, "File bukan gambar yang valid."
    gray = preprocess.to_gray(img)
    box = detector.detect_face(gray)
    if box is None:
        return None, "Wajah tidak terdeteksi. Pastikan wajah menghadap kamera dan cukup terang."
    return preprocess.normalize(preprocess.crop_face(gray, box)), None


@bp.post("/enroll")
def enroll():
    member_id = request.form.get("member_id", "")
    # Laravel mengirim banyak frame dengan nama field "photos".
    # "photo" tetap diterima supaya klien lama (satu foto) tidak rusak.
    files = request.files.getlist("photos") or request.files.getlist("photo")

    if not member_id.isdigit():
        return jsonify(error="member_id tidak valid."), 422
    if not files:
        return jsonify(error="Foto tidak ditemukan."), 422

    # Seperti registrasi.py: hanya frame yang wajahnya terdeteksi Haar cascade yang dipakai.
    # Frame tanpa wajah dilewati, bukan menggagalkan seluruh pendaftaran.
    samples, used = [], 0
    for f in files:
        face, err = _prepare_face(f)
        if err:
            continue
        used += 1
        samples.extend(preprocess.augment(face) if AUGMENT else [face])

    required = min(MIN_VALID_FRAMES, len(files))
    if used < required:
        return jsonify(error=(
            f"Wajah hanya terdeteksi di {used} dari {len(files)} foto (minimal {required}). "
            "Pastikan wajah terlihat jelas, cukup terang, dan menghadap kamera lalu ambil foto ulang."
        )), 422

    saved = dataset.save_samples(int(member_id), samples)
    total = lbph.train()
    return jsonify(member_id=int(member_id), frames=len(files), faces_used=used,
                   samples=saved, trained_images=total)


@bp.delete("/members/<int:member_id>")
def remove_member(member_id):
    dataset.delete_member(member_id)
    return jsonify(trained_images=lbph.train())


@bp.post("/recognize")
def recognize():
    file = request.files.get("photo")
    if file is None:
        return jsonify(error="Foto tidak ditemukan."), 422

    face, err = _prepare_face(file)
    if err:
        return jsonify(error=err), 422

    result = lbph.predict(face)
    if result is None:
        return jsonify(error="Model belum dilatih."), 503

    member_id, distance = result
    matched = distance <= UNKNOWN_THRESHOLD
    return jsonify(matched=matched, member_id=member_id if matched else None,
                   distance=round(distance, 2))


@bp.post("/retrain")
def retrain():
    return jsonify(trained_images=lbph.train())


@bp.get("/health")
def health():
    return jsonify(status="ok")