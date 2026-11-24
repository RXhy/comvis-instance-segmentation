"""Rename foto baru di folder mycapture menjadi mycapture_NN.jpg dan catat di rename_log.csv.

Foto yang sudah bernama mycapture_NN.jpg dilewati. Foto baru diurutkan berdasarkan nama file
(IMG_YYYYMMDD_HHMMSS = urutan waktu foto) lalu diberi nomor lanjutan dari nomor terakhir.

Cara pakai (dari folder tugas_3):
    python rename_mycapture.py
"""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "mycapture")
LOG = os.path.join(HERE, "rename_log.csv")
PATTERN = re.compile(r"^mycapture_(\d+)\.jpg$")
EXTS = (".jpg", ".jpeg")


def taken_time(name):
    m = re.match(r"IMG_(\d{8})_(\d{6})", name)
    if not m:
        return ""
    d, t = m.groups()
    return f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}"


def main():
    files = os.listdir(FOLDER)
    used = [int(m.group(1)) for f in files if (m := PATTERN.match(f))]
    new = sorted(f for f in files if f.lower().endswith(EXTS) and not PATTERN.match(f))
    if not new:
        print("Tidak ada foto baru.")
        return

    next_idx = max(used, default=0) + 1
    width = max(2, len(str(next_idx + len(new) - 1)))
    rows = []
    for i, old in enumerate(new, start=next_idx):
        target = f"mycapture_{i:0{width}d}.jpg"
        if os.path.exists(os.path.join(FOLDER, target)):
            raise SystemExit(f"Batal: {target} sudah ada.")
        rows.append([i, target, old, taken_time(old), "", ""])

    for _, target, old, *_ in rows:
        os.rename(os.path.join(FOLDER, old), os.path.join(FOLDER, target))
        print(f"{old} -> {target}")

    new_log = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["index", "nama_file", "nama_asli", "waktu_foto", "deskripsi", "kondisi"])
        w.writerows(rows)
    print(f"{len(rows)} foto di-rename, log ditambahkan ke rename_log.csv")


if __name__ == "__main__":
    main()
