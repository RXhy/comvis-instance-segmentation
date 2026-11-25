# Logbook Tugas 3 – Segmentasi Instance

| | |
|---|---|
| **Mata kuliah** | EA234503 Visi Komputer |
| **Dataset latih** | Panoptic Indoor Segmentation **v5** (Roboflow Universe, `panopticindoor/panoptic-indoor-segmentation`, CC BY 4.0) |
| **Test set sendiri** | `room_mycapture` (Roboflow): 59 foto, 25 kondisi sulit |
| **Variabel NRP (digit terakhir 2)** | jumlah data latih **25% / 50% / 100%** |
| **Model** | YOLO11s-seg (one-stage CNN), Mask R-CNN R50-FPN v2 (two-stage CNN), Mask2Former Swin-Tiny (transformer) |
| **Tempat komputasi** | Training di Kaggle (GPU T4), inferensi dan analisis di laptop (RTX 3050 6 GB) |

---

## 1. Ringkasan progres

| No | Tahap | Status | Selesai | Bukti / output |
|---|---|---|---|---|
| 1 | Memahami tugas dan statistik dataset v5 | Selesai | 17 Nov | `01_download_dataset.ipynb`, `outputs/class_stats_v5.csv` |
| 2 | Pembagian data dan protokol eksperimen | Selesai | 17 Nov | bagian 2 logbook ini |
| 3 | Pengambilan foto test set sendiri | Selesai | 21 Nov | `rename_log.csv`, `bukti/exif_mycapture.csv` |
| 4 | Hipotesis awal (kualitas dataset dan domain shift) | Dicatat | 17 Nov (ditulis 24 Nov) | bagian 3 logbook ini |
| 5 | Anotasi di Roboflow | Selesai | 24 Nov | `annotation_checklist.md`, `bukti/roboflow_*.png` |
| 6 | Notebook training dan uji kode lokal | Selesai | 23 Nov | `02_*.ipynb`, `03_*.ipynb`, `04_*.ipynb` |
| 7 | Training 3 model × 3 fraksi di Kaggle | Selesai | 24 Nov | `*/metrics.csv`, `*/gpu_log.csv`, `*/<run>/results.csv` |
| 8 | Hipotesis perbandingan model | Dicatat | 24 Nov | bagian 3 logbook ini |
| 9 | Evaluasi mycapture (domain shift) | Selesai | 24 Nov | `hasil_lokal/metrics_mycapture.csv` |
| 10 | Analisis 5 foto gagal + eksperimen kecil | Selesai | 24 Nov | `05_inferensi_evaluasi.ipynb` bagian 7, `hasil_lokal/eksperimen/` |
| 11 | Laporan dan notebook inferensi | Selesai | 25 Nov | `laporan.md`, `05_inferensi_evaluasi.ipynb` |
| 12 | Repositori Git (riwayat commit) | Berjalan | 25 Nov | GitHub `comvis-instance-segmentation` |
| 13 | Notebook demo live | Selesai | 25 Nov | `06_demo.ipynb` |
| 14 | Administrasi (nama/NRP, blur voucher, izin wajah) | Selesai | 25 Nov | lihat bagian 6 dan 7 |

---

## 2. Logbook mingguan

### Minggu 1 (17–23 Nov 2026)

| No | Tanggal | Kegiatan | Hasil | Kendala | Keputusan / solusi | Bukti |
|---|---|---|---|---|---|---|
| 1 | 17 Nov | Membaca slide tugas (PDF pertemuan 10) dan membuat notebook download v5 (COCO segmentation) | Daftar poin wajib: ≥3 model dari ≥2 keluarga, test set sendiri ≥40 foto (≥10 sulit), domain shift, 5 analisis kegagalan, log, GPU, logbook, deklarasi AI | Download gagal: `A valid API key must be provided`. Ternyata variabel `API_KEY` kosong (key tidak ter-paste, urutan cell terbalik) | Cell API key diberi pengecekan: berhenti kalau kosong dan menampilkan panjang key yang terbaca | `01_download_dataset.ipynb` |
| 2 | 17 Nov | Menghitung statistik per kelas v5 | 397 / 166 / 359 gambar, **74 kelas**, 8 kelas tanpa contoh train, maks. 129 objek per gambar (detail di Lampiran A) | Halaman project menampilkan 93 kelas, termasuk typo; distribusi train dan test sangat berbeda | Daftar kelas resmi diambil dari file anotasi v5. Ketidakcocokan train–test dicatat sebagai **domain shift bawaan dataset** | `outputs/class_stats_v5.csv`, `outputs/class_names_v5.txt` |
| 3 | 17 Nov | Merencanakan pembagian data | Protokol eksperimen | Ide awal: pilih model terbaik di test v5, lalu hanya model itu yang dites ke foto sendiri | Ide awal dibatalkan karena tidak sesuai instruksi dan membuat test set ikut memilih model. 25/50/100% **hanya untuk split train**; valid dan test v5 tetap utuh; checkpoint dipilih dari **valid**; **semua model** dites di kedua test set; foto sendiri tidak ikut training | – |
| 4 | 17 Nov | Mengambil foto pertama dengan HP (POCO X7 Pro, 4096×3072, rasio 4:3 seperti v5) | 43 foto rumah + 5 foto malam yang lebih sederhana | Banyak foto gudang sangat penuh (#20–#23), banyak sudut yang mirip, belum ada `person` dan kondisi redup. Satu foto memuat voucher dengan barcode | Foto di-rename `mycapture_NN.jpg` (urut waktu), nama asli dicatat di `rename_log.csv`. Benda tanpa kelas **dibiarkan tanpa label**. Voucher harus di-blur sebelum dilampirkan | `rename_log.csv` |
| 5 | 17 Nov | Menyusun hipotesis awal | 3 dugaan: dataset jelek, data timpang, mycapture beda jauh (bagian 3) | – | – | bagian 3 |
| 6 | 18–21 Nov | Membuat project Roboflow `room_mycapture` (Instance Segmentation) dan mulai anotasi dari #04 | Aturan anotasi tertulis (Lampiran B) | Bingung tampilan polygon vs "mask" (pratinjau menambah kotak otomatis) | Anotasi memakai **polygon**, sama dengan format v5. Kotak hanya turunan otomatis | `annotation_checklist.md` |
| 7 | 18–21 Nov | Anotasi lanjutan | – | Alat Bounding Box aktif tanpa sengaja; dinding keramik #16 sempat diberi label `wall-other-merged`; tidak ada cara menambah vertex ke polygon yang sudah jadi | Selalu cek alat Polygon/Smart Polygon yang aktif; #16 diperbaiki menjadi `wall-tile`; vertex diedit dengan Drag/Select atau polygon digambar ulang | – |
| 8 | 19 & 21 Nov | Menambah 13 foto: kafe, lab komputer, ruang kelas, ballroom, termasuk foto dengan orang | Total **61 foto**, rename otomatis | File bernama hanya `.jpg` (tersembunyi) ikut di-rename menjadi `mycapture_49.jpg` | Isi file dicek: foto asli dari HP. Waktu foto diisi dari EXIF (18 Nov 12:34) | `rename_mycapture.py`, `rename_log.csv` |
| 9 | 23 Nov | Memilih 3 model | Tabel pilihan model (Lampiran C) | GPU lokal hanya RTX 3050 6 GB | Training resmi di **Kaggle T4** | – |
| 10 | 23 Nov | Membuat notebook training 02–04 | Subset bertingkat 25% ⊂ 50% ⊂ 100% (seed 42), batas **40 menit per run**, LR mengikuti batas waktu, evaluasi pycocotools `MAX_DET=200` | – | Protokol sama untuk ketiga model (Lampiran C) | `02_train_yolo11seg.ipynb`, `03_train_maskrcnn.ipynb`, `04_train_mask2former.ipynb` |
| 11 | 23 Nov | Uji kode lokal: smoke test (12 gambar, 1 epoch) dan uji overfit | Semua notebook jalan. Overfit 12 gambar: Mask R-CNN AP 0,53, Mask2Former 0,97 → pipeline dan label benar | AP@[.50:.95] selalu 0 padahal AP50 0,33 (kasus AI no. 1). Beberapa epoch lambat saat GPU laptop 86 °C | AP dihitung langsung dari tabel `precision` dan dicocokkan dengan pycocotools (0,2158 vs 0,2158). Throttling lokal diabaikan karena training resmi di Kaggle | – |
| 12 | 23 Nov | Training YOLO dan Mask R-CNN di Kaggle | Mask AP test v5: YOLO 0,138 / 0,092 / 0,089; Mask R-CNN 0,062 / 0,090 / 0,105 | Split mycapture dilewati karena versi Roboflow belum ada. **YOLO turun** saat data ditambah | Bobot disimpan, mycapture dievaluasi belakangan tanpa training ulang. Penurunan YOLO dianalisis sebagai *catastrophic forgetting* (Lampiran D) | `yolo11s/`, `maskrcnn/` |
| 13 | 23 Nov | Training Mask2Former percobaan 1 | Gagal | **CUDA out of memory** saat evaluasi test v5: gambar 3024×3398, 50 mask sekaligus = 3,8 GB (kasus AI no. 5) | Jumlah mask per potongan disesuaikan ukuran gambar. Diuji di 4 gambar test terbesar: puncak memori 1,06 GB | `04_train_mask2former.ipynb` |

### Minggu 2 (24 Nov 2026 – …)

| No | Tanggal | Kegiatan | Hasil | Kendala | Keputusan / solusi | Bukti |
|---|---|---|---|---|---|---|
| 14 | 23–24 Nov | Training Mask2Former percobaan 2 | Mask AP test v5: 0,146 / 0,164 / 0,165, **terbaik di semua fraksi** | AP valid run 100% < run 50%. `results.csv` Mask R-CNN dan Mask2Former tidak bisa dibaca pandas (kasus AI no. 6) | Run 100% hanya sempat 28 epoch, iterasinya hampir sama dengan run 50%, jadi dengan waktu tetap data 2× tidak menambah AP. `append_csv` diperbaiki dan 6 file diperbaiki header-nya, isi data tidak diubah | `mask2former/` |
| 15 | 24 Nov | Menulis hipotesis perbandingan model | 3 dugaan: Mask2Former terbaik, Mask R-CNN terburuk, wall/chair/person bagus (bagian 3) | Angka test v5 sudah sempat terlihat di percakapan dengan AI | Hipotesis hanya diuji pada **mycapture** yang belum dievaluasi | bagian 3 |
| 16 | 24 Nov | Menghapus foto #60 dan #61 (ballroom, redup) | Test set **59 foto**, **25 kondisi sulit** (syarat ≥40 dan ≥10 tetap terpenuhi) | Resolusi kedua foto berbeda dari foto lain. Kondisi `redup` tinggal 1 foto (#56) | Baris #60/#61 dihapus dari `rename_log.csv` supaya nomor 60 bisa dipakai lagi. Analisis kondisi "redup" dinyatakan lemah | `rename_log.csv`, `annotation_checklist.md` |
| 17 | 24 Nov | Mengisi kolom kondisi tiap foto | 34 normal, 25 sulit: penuh dan/atau saling menutupi 16, backlight 7, sudut ekstrem 1, redup 1 (#56, juga penuh) | – | – | `rename_log.csv` |
| 18 | 24 Nov | Memutuskan label kardus | Semua kardus dan kontainer bertutup dilabeli **`box`** | v5 punya `box` (train 95) dan `cardboard` (train 2) | `box` dipilih supaya yang terukur kemampuan model, bukan ketidakcocokan label. Temuan sampingan: v5 tidak konsisten antar-split (Lampiran E) | `outputs/class_stats_v5.csv` |
| 19 | 24 Nov | Anotasi selesai, versi dataset dibuat di Roboflow (15.49) | 59 gambar, 871 anotasi, 55 kelas dipakai (Lampiran F) | Split tidak bisa diubah ke 100% test → **0 / 2 / 57**. Daftar kelas tertulis 76 (ada sisa typo `ceilig`) | Notebook menggabungkan semua split mycapture menjadi satu test set. Kelas yang tidak dipakai tidak berpengaruh | `bukti/roboflow_split_2026-11-24.png`, `bukti/roboflow_analytics_2026-11-24.png` |
| 20 | 24 Nov | Mengecek file export (COCO Segmentation, 640×480) | 898 polygon (lebih banyak dari screenshot karena ada perbaikan anotasi setelahnya) | Ada kelas di luar v5: `chair-crowd` (12 polygon, disengaja) dan `nox` (1, typo dari `box` di #46) | `nox` → `box` saat evaluasi (`MYCAP_MERGE`). `<kelas>-crowd` dibaca sebagai `<kelas>` dengan `iscrowd=1` (area abaikan) | `data/room_mycapture.v1i.coco-segmentation` |
| 21 | 24 Nov | Memindahkan kode inferensi ke `lib/` dan uji reproduksi di laptop | Mask AP test v5 laptop vs Kaggle: YOLO 0,0891 vs 0,0890; Mask R-CNN 0,1049 vs 0,1049; Mask2Former 0,1651 vs 0,1651 | – | Bobot dan kode inferensi lokal terbukti sama dengan saat training, jadi evaluasi mycapture dilakukan di laptop | `lib/common.py`, `lib/models.py`, `lib/analysis.py` |
| 22 | 24 Nov | Evaluasi 3 model × 3 fraksi di mycapture | AP mycapture < test v5 di **9 dari 9 run** (−14% s.d. −32%) (Lampiran D) | – | – | `hasil_lokal/metrics_mycapture.csv`, `hasil_lokal/grafik_*.png` |
| 23 | 24 Nov | Analisis 5 foto gagal + eksperimen kecil | #55 kursi → `stairs`; #51 keyboard/mouse terlupakan; #28 piring → `pan`; #41 backlight + gamma; #49 batas mask 28×28 (Lampiran G) | – | Setiap penjelasan diuji dengan eksperimen; hipotesis resolusi mask di #49 **terbantah** | `05_inferensi_evaluasi.ipynb` bagian 7, `hasil_lokal/eksperimen/` |
| 24 | 24 Nov | Membuat notebook laporan utama | Hipotesis, tahapan, inferensi, hasil, pembahasan, analisis kegagalan, bukti | Font markdown notebook terlalu besar | Ukuran font diatur lewat setting VS Code | `05_inferensi_evaluasi.ipynb` |
| 25 | 24 Nov | Menyiapkan bukti EXIF dan `.gitignore` | EXIF 59 foto tanpa GPS; screenshot detail foto di HP | EXIF semua foto berisi **GPS alamat rumah**; ada wajah orang | Folder `mycapture/` tidak di-commit. EXIF dibagikan tanpa GPS; screenshot dipotong di atas baris lokasi | `bukti/exif_mycapture.csv`, `bukti/exif_hp_mycapture_01.png`, `.gitignore` |
| 26 | 25 Nov | Memisahkan laporan dari notebook | Laporan ditulis ulang sebagai `laporan.md` (tanpa kode, gambar diambil dari `hasil_lokal/`); notebook 05 dijadikan notebook inferensi dan evaluasi saja | Notebook 05 bercampur antara kode, output, dan narasi panjang, sehingga sulit dibaca sebagai laporan | Narasi dipindah ke `laporan.md`; `05_laporan_utama.ipynb` diganti nama menjadi `05_inferensi_evaluasi.ipynb`, kode dan output tidak diubah | `laporan.md`, `05_inferensi_evaluasi.ipynb` |
| 27 | 25 Nov | Demo live dan administrasi | Demo inferensi foto baru sudah dilakukan saat sesi; izin dari semua orang yang wajahnya ada di foto (#51, #53, #54, #57, #58) sudah didapat | – | Foto #01 (voucher) tidak perlu di-blur karena tidak dilampirkan di mana pun | `06_demo.ipynb` |

---

## 3. Hipotesis (bertanggal)

### 3.1 Hipotesis awal – 17 Nov 2026

> Disampaikan saat pengambilan foto, **sebelum training**. Baru dicatat ke logbook pada 24 Nov 2026.

Dugaan saya: secara keseluruhan **semua model akan berperforma buruk**, apa pun arsitekturnya.

| No | Hipotesis | Alasan | Status (24 Nov) | Bukti |
|---|---|---|---|---|
| 1 | Kualitas dataset v5 rendah | Train hanya 397 gambar untuk 74 kelas, ada kelas typo/duplikat (`scissor`/`scissors`, `wall`/`wall-other-merged`, `st-`), dan 8 kelas tanpa contoh train | **Terdukung** | Mask AP test v5 tertinggi hanya 0,165 (Mask2Former 100%); YOLO 100% hanya 0,089. Sebagai pembanding kasar, mask AP ketiga model di COCO sekitar 0,38–0,45. Kelas tanpa contoh train (`keyboard`, `mouse`, `scissors`) hanya dikenali YOLO yang masih "ingat" COCO |
| 2 | Data timpang, sehingga AP test v5 jauh di bawah AP valid | Antar-kelas: `wall-other-merged` 756 instance, banyak kelas hanya 1–5. Antar-split: handbag 5 vs 125, dining table 4 vs 112, wallet 197 vs 5 | **Terdukung** | Gap valid vs test v5 besar di ketiga model (Mask R-CNN 0,187 vs 0,105; Mask2Former 0,289 vs 0,165). Ada juga inkonsistensi label: kardus = `box` di train, `cardboard` di test |
| 3 | mycapture sangat berbeda dari train, sehingga **AP mycapture < AP test v5** untuk semua model | Foto dari rumah, kafe, kelas, dan ballroom Indonesia, banyak kondisi sulit dan benda tanpa kelas | **Terdukung** | 9 dari 9 run (−14% s.d. −32%); `05_inferensi_evaluasi.ipynb` bagian 5.3 |

### 3.2 Hipotesis perbandingan model – 24 Nov 2026

> Ditulis **setelah** training dan evaluasi test v5 selesai (23 Nov), tapi **sebelum** evaluasi mycapture dan analisis
> kegagalan. Ringkasan angka test v5 sudah sempat muncul di percakapan dengan AI, jadi hipotesis ini hanya diuji pada **mycapture**.
> Cara menguji: urutan mask AP ketiga model di mycapture, dan AP per kelas mycapture (run 100%).

| No | Hipotesis (kata-kata saya) | Alasan arsitektur / data | Status | Bukti |
|---|---|---|---|---|
| 1 | **Mask2Former terbaik** | Transformer dan dirancang khusus untuk segmentasi; seri YOLO lebih mengutamakan kecepatan inferensi (real-time) | **Terdukung** | Terbaik di semua fraksi (mycapture 0,122 / 0,135 / 0,140) |
| 2 | **Mask R-CNN terburuk** | – | **Sebagian** | Benar di 25% dan 50%, salah di 100%: YOLO 0,070 < Mask R-CNN 0,090 karena YOLO terkena *catastrophic forgetting* |
| 3a | `wall-other-merged` bagus | Porsinya paling banyak di data train | **Terdukung** | Peringkat 3 dari 53 kelas (AP rata-rata 0,415) |
| 3b | `person` bagus | Contohnya di train jauh lebih banyak dari jumlahnya di test saya | **Terdukung** | Peringkat 4 dari 53 kelas (0,399) |
| 3c | `chair` bagus | Sama dengan 3b | **Tidak terdukung** | Peringkat 30, AP 0,013–0,048. Barisan kursi ballroom/kelas dikira `stairs`, dan YOLO melupakan kursi COCO |

---

## 4. Catatan waktu GPU

| Tanggal | Tempat | Model | Run | Menit | Keterangan |
|---|---|---|---|---|---|
| 23 Nov 2026 | Lokal RTX 3050 | ketiga model | smoke test | < 10 per model | uji kode, bukan hasil resmi |
| 23 Nov 2026 | Kaggle T4 | YOLO11s-seg | 025 / 050 / 100pct | train 3,7 / 11,2 / 22,8; eval 0,7 / 0,7 / 0,8 | total ≈ 40 menit; early stop di epoch 33 / 88 / 114 (`yolo11s/gpu_log.csv`) |
| 23 Nov 2026 | Kaggle T4 | Mask R-CNN | 025 / 050 / 100pct | train 16,4 / 23,9 / 40,3; eval 2,0 / 1,9 / 1,7 | total ≈ 86 menit; 40 epoch penuh, best epoch 30 / 32 / 32 (`maskrcnn/gpu_log.csv`) |
| 23 Nov 2026 | Kaggle T4 | Mask2Former | percobaan 1 | 113 (1 jam 53 menit) | gagal: CUDA OOM saat evaluasi test v5 (kasus AI no. 5); waktu dari riwayat run Kaggle ("Failed after 1 hour and 53 minutes") |
| 23 Nov 2026 | Kaggle T4 | Mask2Former | 025 / 050 / 100pct | train 24,8 / 39,4 / 40,5; eval 1,8 / 1,7 / 2,0 | total ≈ 110 menit; 55 / 50 / 28 epoch, best epoch 35 / 30 / 15 (`mask2former/gpu_log.csv`) |
| 24 Nov 2026 | Lokal RTX 3050 | ketiga model | inferensi mycapture + eksperimen | 3–15 detik per run (59 foto) | `05_inferensi_evaluasi.ipynb` bagian 4 |

---

## 5. Deklarasi penggunaan AI

AI (Claude, lewat Claude Code) dipakai untuk:
- memahami tugas dan menganalisis statistik dataset;
- menyusun rencana foto dan panduan anotasi Roboflow;
- memilih model;
- menulis kode notebook (download data, subset, training, evaluasi, laporan), modul `lib/`, dan script rename.

Seluruh pengambilan foto, anotasi, dan keputusan akhir dikerjakan sendiri.

### 5.1 Prompt utama

| No | Prompt (ringkas) | Dipakai untuk |
|---|---|---|
| 1 | "aku mau tau classnya apa aja biar aku bisa ambil foto … buat notebook untuk download" | notebook 01 dan daftar kelas |
| 2 | "itu udah ada outputnya, interpretasiin … wall other merged apaan" | memahami statistik v5 |
| 3 | "kasih saran apa aja yang harus difoto" | rencana foto |
| 4 | "apakah fotoku terlalu sulit" | evaluasi komposisi foto |
| 5 | "cara annotate di roboflow gimana" / "pake polygon aman kan" | panduan anotasi |
| 6 | "saran modelnya apa aja" | pemilihan model |
| 7 | "buatin notebook untuk training 3 modelnya, aku mau train di kaggle" | notebook 02–04 |
| 8 | "kamu bisa tau objek di mycapture apa aja, dan yang tidak ada kelasnya" | `annotation_checklist.md` |

### 5.2 Kasus saran/kode AI yang salah dan perbaikannya

| No | Kasus | Cara ketahuan | Perbaikan |
|---|---|---|---|
| 1 | **AP@[.50:.95] selalu 0.** Kode evaluasi dari AI mengambil `COCOeval.stats`. Karena batas deteksi diubah menjadi 200, fungsi ringkasan pycocotools (yang selalu mencari `maxDets=100`) mengembalikan −1 → tercatat 0 | Smoke test: AP50 0,33 tapi AP 0,00, tidak masuk akal | AP dihitung langsung dari tabel `precision` hasil `accumulate()`, dicocokkan dengan pycocotools pada `maxDets=100` → identik (0,2158 vs 0,2158) |
| 2 | **Daftar kelas awal tidak akurat.** Daftar 93 kelas dari AI diambil dari halaman project Roboflow dan memuat typo | Setelah file COCO v5 dianalisis, v5 hanya punya 74 kelas | Daftar resmi diambil dari file anotasi v5 (`outputs/class_names_v5.txt`) |
| 3 | **`!pip install` di notebook lokal** bisa memasang package ke Python yang berbeda dari kernel | – | Diganti `%pip install` |
| 4 | **Script rename ikut mengganti file tersembunyi** bernama `.jpg` menjadi `mycapture_49.jpg` | File tidak terlihat saat pengecekan awal, baru ketahuan setelah rename | Isi file dicek (foto asli HP, EXIF 18 Nov 12:34), waktu foto dilengkapi manual di `rename_log.csv` |
| 5 | **Mask2Former CUDA out of memory.** Fungsi prediksi dari AI memperbesar 50 mask sekaligus ke resolusi asli; gambar test v5 sampai 3024×3398 butuh 3,8 GB | Run Kaggle gagal; smoke test lokal tidak menangkap karena gambarnya kecil | Jumlah mask per potongan disesuaikan ukuran gambar; puncak memori di 4 gambar terbesar 1,06 GB |
| 6 | **Log training per epoch rusak.** `append_csv` dari AI menulis header dari baris pertama saja; kolom AP valid baru muncul di epoch evaluasi | `results.csv` tidak bisa dibaca pandas | Fungsi menulis ulang header kalau ada kolom baru; file lama diperbaiki dengan menambah 2 nama kolom di header, isi data tidak diubah |

---

## 6. Keputusan yang sempat tertunda

| No | Hal | Status | Keputusan |
|---|---|---|---|
| 1 | `CLASS_MERGE` (`scissor→scissors`, `wall→wall-other-merged`, buang `st-`; total 5 anotasi) | Diputuskan | **Tidak dipakai.** Semua run memakai `CLASS_MERGE = {}`, sama di ketiga notebook, sehingga perbandingan tetap adil |
| 2 | Mengurangi foto ballroom (#55, #56, #59) dan gudang (#20–#23) yang berat dianotasi | Diputuskan | Tetap dipakai dan dianotasi lengkap. Yang dihapus hanya #60 dan #61 karena resolusinya berbeda |
| 3 | Izin dari orang yang wajahnya ada di foto (#51, #53, #54, #57, #58) | Diputuskan | Semua orang di foto tersebut sudah memberi izin (25 Nov) |

---

## 7. Rencana berikutnya

| No | Kegiatan | Status | Catatan |
|---|---|---|---|
| 1 | Membuat repositori Git privat dan commit bertahap | Berjalan | repo `comvis-instance-segmentation` sudah ada; commit dan push lanjutan oleh saya sendiri |
| 2 | Notebook demo live (inferensi foto baru, ubah threshold, AP satu kelas) | Selesai (25 Nov) | `06_demo.ipynb`; demo sudah dilakukan saat sesi |
| 3 | Mengisi nama dan NRP di laporan | Selesai | `laporan.md` |
| 4 | Blur voucher/barcode di foto #01 sebelum dilampirkan | Tidak perlu | Foto #01 tidak dilampirkan di laporan maupun repo (folder `mycapture/` tidak di-commit) |
| 5 | Mengisi menit GPU Mask2Former percobaan 1 | Selesai | 113 menit, dari riwayat run Kaggle |
| 6 | Menyimpan bobot model di Google Drive (opsional) | Belum | bobot >100 MB, tidak bisa masuk GitHub |

---

## Lampiran – catatan detail

### A. Statistik dataset v5 (`outputs/class_stats_v5.csv`)

| Hal | Nilai |
|---|---|
| Jumlah gambar | train 397, valid 166, test 359 |
| Jumlah anotasi | train 4.479, valid 2.022, test 5.344 |
| Jumlah kelas | **74** (halaman project menampilkan 93, termasuk typo yang tidak ada di v5) |
| Kelas tanpa contoh di train | 8: toothbrush, mouse, keyboard, flower, scissors, teddy bear, toaster, hair drier |
| Instance terbanyak per gambar | 129 (di split test) |
| Ketidakcocokan train vs test | handbag 5 vs 125, dining table 4 vs 112, curtain 3 vs 107, wallet 197 vs 5, trash-bin 153 vs 4, box 95 vs 1 |
| Sisa masalah label | `st-` (1), `scissor` vs `scissors`, `wall` vs `wall-other-merged`, `paper` vs `paper-merged` |

Kelas `*-merged` dan `*-stuff` adalah kelas "stuff" dari COCO-Panoptic (mis. `wall-other-merged` = dinding cat/plester biasa).

### B. Aturan anotasi

| Jenis | Aturan |
|---|---|
| **Things** (chair, person, bottle, bed, …) | Satu objek = satu anotasi, walaupun terpotong benda lain. Penghalang tipis → satu polygon melewati penghalang; penghalang besar → Brush/mask satu anotasi, atau bagian terlihat terbesar saja. Area kerumunan kursi yang tidak bisa dipisah → `chair-crowd` |
| **Stuff** (wall, floor, ceiling, rug) | Satu polygon per area yang tersambung; kalau terpotong benda sampai terpisah → polygon terpisah (sesuai pola v5: lebih dari separuh gambar train memecah `wall-other-merged` menjadi ≥3 potongan) |
| Bagian terlihat | Hanya bagian yang **terlihat** yang dianotasi; tumpang tindih kecil di tepi tidak masalah |
| Satu kelas | Satu objek hanya punya satu kelas (tidak boleh label ganda) |
| Benda di dinding | Dipotong dari polygon dinding. Kalau ada kelasnya (bottle, clock, mirror-stuff, plug, light) dianotasi sendiri; kalau tidak (rak sabun, poster, lukisan) dibiarkan. Benda sangat kecil (paku, saklar) boleh ikut dinding |
| Kelas typo | `cuo`, `cyo`, dll. **tidak dipakai**; nama kelas selalu dicek ke `outputs/class_names_v5.txt` |
| Bantuan AI | **Tidak memakai** "Find Objects with AI" / Label Assist, supaya ground truth tidak bias ke prediksi model. Smart Polygon (SAM) hanya untuk menggambar tepi; kelas dan objek ditentukan sendiri |

**Pemetaan benda ambigu**

| Benda | Kelas |
|---|---|
| stool / bangku | `chair` |
| bangku kayu berbantal | `couch` |
| monitor | `tv` |
| tablet | `cell phone` |
| lemari laci plastik, loker, kabinet dapur | `cabinet-merged` |
| rak terbuka | `shelf` |
| toples, galon | `bottle` |
| kardus, kontainer plastik/styrofoam | `box` (bukan `cardboard`, lihat Lampiran E) |
| meja makan bundar | `dining table` |
| backsplash mozaik/panel, dinding keramik | `wall-tile` |
| karpet ballroom | `rug-merged` |
| pakaian, kresek, keranjang | tanpa label |

### C. Pilihan model dan protokol eksperimen

| Model | Keluarga | Library | Alasan |
|---|---|---|---|
| YOLO11s-seg | one-stage CNN | Ultralytics | cepat; mask dari *prototype* resolusi rendah (relevan untuk analisis kegagalan) |
| Mask R-CNN R50-FPN v2 | two-stage CNN | torchvision | mask head 28×28 per RoI (disebut di materi); mudah dipakai di Windows/Kaggle |
| Mask2Former Swin-Tiny | transformer | Hugging Face | tanpa proposal/box, mask dari *query*; pembanding arsitektur |

| Aspek | Protokol |
|---|---|
| Subset | Bertingkat 25% ⊂ 50% ⊂ 100% (99 / 198 / 397 gambar), seed 42; gambar berkelas langka diambil lebih dulu |
| Inisialisasi | Bobot pretrained COCO; input sisi panjang 640 |
| Waktu | 40 menit per run (3 run × 40 menit = 2 jam GPU per model); jadwal LR menyesuaikan batas waktu |
| Pemilihan checkpoint | Mask AP terbaik di `valid` |
| Evaluasi | pycocotools (mask AP & box AP), `MAX_DET=200`, `SCORE_THRESH=0.05`, di valid, test v5, dan mycapture |
| Kelas mycapture | Dipetakan ke 74 kelas v5 berdasarkan nama; nama tidak dikenal dibuang dan dilaporkan |
| Perbedaan antar model | Ultralytics otomatis memakai ulang kepala klasifikasi COCO untuk 33 kelas yang namanya sama; Mask R-CNN dan Mask2Former mulai kepala klasifikasi dari nol. Augmentasi mengikuti resep bawaan masing-masing |

### D. Ringkasan hasil (mask AP)

| Split | YOLO 25 / 50 / 100% | Mask R-CNN 25 / 50 / 100% | Mask2Former 25 / 50 / 100% |
|---|---|---|---|
| valid | 0,129 / 0,187 / 0,212 | 0,120 / 0,158 / 0,186 | 0,266 / 0,314 / 0,288 |
| test v5 | 0,138 / 0,092 / 0,089 | 0,062 / 0,090 / 0,105 | 0,146 / 0,164 / 0,165 |
| mycapture | 0,107 / 0,065 / 0,070 | 0,044 / 0,061 / 0,090 | 0,122 / 0,135 / 0,140 |

**Temuan *catastrophic forgetting* YOLO (23 Nov).** Run YOLO 25% punya best epoch = 3, jadi isinya hampir model COCO
pretrained. AP rata-rata di test v5 untuk kelas COCO turun 0,30 → 0,08 (25% → 100%), sedangkan kelas non-COCO naik
0,00 → 0,10. Penurunan terbesar di `mouse`, `keyboard`, dan `scissors`, padahal ketiganya tidak punya contoh train di v5.
Mask2Former tidak mengalami ini (kelas COCO / non-COCO 0,16 / 0,13 → 0,17 / 0,16).

### E. Data keputusan `box` vs `cardboard`

| Kelas | Instance train | Valid | Test |
|---|---|---|---|
| `box` | 95 | 85 | 1 |
| `cardboard` | 2 | 0 | 14 |

`cardboard` hanya punya 2 contoh train, jadi model praktis tidak pernah memprediksinya. Kalau kardus dilabeli `cardboard`,
AP kelas itu otomatis ≈ 0 dan prediksi `box` yang secara visual benar ikut dihitung salah. `box` mengikuti kebiasaan label
di train dan valid v5. Temuan sampingan: v5 sendiri tidak konsisten (kardus = `box` di train/valid, `cardboard` di test),
salah satu penjelasan gap valid vs test v5.

### F. Dataset mycapture di Roboflow (24 Nov, Dataset Analytics)

| Hal | Nilai |
|---|---|
| Jumlah gambar | 59 (0 tanpa anotasi) |
| Jumlah anotasi | 871 (rata-rata 14,8 per gambar); export akhir 898 polygon |
| Kelas yang dipakai | 55 |
| Ukuran asli | 4096×3072 (12,58 MP); export 640×480 |
| Kelas terbanyak | `box` 168, `chair` 100, `bottle` 91, `wall-other-merged` 58 |
| Split | Train 0 / Valid 2 / Test 57 (digabung menjadi satu test set) |

![Dataset Analytics Roboflow](bukti/roboflow_analytics_2026-11-24.png)
![Contoh anotasi mycapture_24](bukti/roboflow_anotasi_mycapture_24.png)

### G. Lima kasus kegagalan dan eksperimennya

| Foto | Kegagalan | Eksperimen | Hasil |
|---|---|---|---|
| #55 | Barisan kursi diprediksi `stairs` | Crop + zoom | Tidak menolong → masalah klasifikasi (stairs punya 175 instance train), bukan ukuran |
| #51 | `keyboard`/`mouse` tidak terdeteksi | Bandingkan fraksi 25/50/100% | Hanya YOLO 25% yang mengenali → *catastrophic forgetting* kelas COCO |
| #28 | Piring diprediksi `pan` | Crop + zoom (±3,7×) | Mask2Former 6/8 dan Mask R-CNN 3/8 piring benar; YOLO tetap 0 → sebagian masalah resolusi |
| #41 | Backlight | Koreksi gamma 0,5 di 7 foto backlight | YOLO 0,097 → 0,126; Mask R-CNN 0,129 → 0,126; Mask2Former 0,235 → 0,157 |
| #49 | Mask stool tidak rapi | Batas atas grid 28×28; YOLO `imgsz` 640/960/1280 | Batas atas 0,973 vs IoU tercapai 0,31–0,35; `imgsz` lebih besar justru lebih buruk → hipotesis resolusi mask **terbantah** |
