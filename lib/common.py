"""Kode bersama untuk notebook training Tugas 3 (Panoptic Indoor v5).

Isi: download data, penyeragaman kelas, subset train bertingkat (25/50/100%),
evaluasi COCO (pycocotools) yang sama untuk semua model, log waktu GPU, dan overlay prediksi.
File ini ditulis ulang oleh cell %%writefile di setiap notebook, jadi isinya identik di ketiga notebook.
"""
import os, json, time, random, shutil, csv, datetime, math
from collections import defaultdict, Counter

import numpy as np
from PIL import Image

ON_KAGGLE = os.path.exists("/kaggle/working")
ON_WINDOWS = os.name == "nt"


# ----------------------------------------------------------------------------- util umum
def get_secret(name):
    """Ambil secret dari Kaggle Secrets / Colab Secrets / env var / input manual."""
    try:
        from kaggle_secrets import UserSecretsClient
        v = UserSecretsClient().get_secret(name)
        if v:
            return v.strip()
    except Exception:
        pass
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v:
            return v.strip()
    except Exception:
        pass
    if os.environ.get(name):
        return os.environ[name].strip()
    if ON_KAGGLE:
        raise RuntimeError(f"Secret '{name}' tidak ditemukan. Tambahkan lewat Add-ons -> Secrets, "
                           "lalu centang/attach ke notebook ini.")
    from getpass import getpass
    return getpass(f"{name}: ").strip()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def append_csv(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows, fields = [], []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields, rows = list(reader.fieldnames or []), list(reader)
    if all(k in fields for k in row):
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields or list(row.keys()))
            if not fields:
                w.writeheader()
            w.writerow(row)
        return
    # ada kolom baru (mis. val_mask_AP baru muncul di epoch evaluasi): tulis ulang dengan header gabungan
    fields += [k for k in row if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows + [row])


def frac_tag(frac):
    return f"{int(round(frac * 100)):03d}pct"


def gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


class GpuTimer:
    """Mencatat waktu GPU per run ke results/gpu_log.csv (bukti proses untuk laporan)."""

    def __init__(self, results_dir, model, tag, note=""):
        self.path = os.path.join(results_dir, "gpu_log.csv")
        self.model, self.tag, self.note = model, tag, note

    def __enter__(self):
        self.t0 = time.time()
        self.start = datetime.datetime.now().isoformat(timespec="seconds")
        return self

    def __exit__(self, *exc):
        self.minutes = (time.time() - self.t0) / 60
        append_csv(self.path, {
            "model": self.model, "run": self.tag, "gpu": gpu_name(),
            "start": self.start, "end": datetime.datetime.now().isoformat(timespec="seconds"),
            "minutes": round(self.minutes, 2), "status": "error" if exc[0] else "ok", "note": self.note,
        })
        return False


# ----------------------------------------------------------------------------- download
def has_coco_split(d):
    return d and os.path.isdir(d) and any(
        os.path.exists(os.path.join(d, s, "_annotations.coco.json")) for s in ("train", "valid", "test"))


def download_roboflow(api_key, workspace, project, version, dest, fmt="coco-segmentation"):
    if has_coco_split(dest):
        print("Sudah ada:", dest)
        return dest
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(fmt, location=dest, overwrite=False)
    return ds.location


# ----------------------------------------------------------------------------- kelas & konversi COCO
def real_categories(coco):
    """Buang kategori induk buatan Roboflow (mis. 'indoor-objects'), sisanya kategori asli."""
    cats = coco["categories"]
    supers = {c.get("supercategory") for c in cats}
    return [c for c in cats if not (c["name"] in supers and c.get("supercategory") in ("none", None, ""))]


def canonical_classes(v5_train_json, merge=None):
    """Daftar kelas final (urutan = urutan kategori v5). merge: {nama_lama: nama_baru atau None(=buang)}."""
    merge = merge or {}
    names = []
    for c in real_categories(load_json(v5_train_json)):
        n = merge.get(c["name"], c["name"])
        if n is not None and n not in names:
            names.append(n)
    return names


def _poly_ok(seg):
    return isinstance(seg, list) and any(len(p) >= 6 for p in seg)


def _resize_rle(seg, h, w, nh, nw):
    from pycocotools import mask as M
    rle = M.frPyObjects(seg, h, w) if isinstance(seg.get("counts"), list) else seg
    m = M.decode(rle)
    m = np.array(Image.fromarray(m * 255).resize((nw, nh), Image.NEAREST)) > 127
    out = M.encode(np.asfortranarray(m.astype(np.uint8)))
    out["counts"] = out["counts"].decode("ascii")
    return out


def to_canonical_coco(src_dirs, classes, out_json, merge=None, resized_dir=None, max_side=1024, target_side=640):
    """Gabung satu/lebih folder split Roboflow (COCO) jadi satu COCO dengan id kelas kanonik 1..K.

    - Nama kelas dipetakan lewat `merge`, kelas yang tidak ada di `classes` dibuang (dilaporkan).
    - Gambar yang sisi panjangnya > max_side diperkecil ke target_side (anotasi ikut diskalakan),
      supaya evaluasi foto HP resolusi penuh tidak kehabisan memori.
    - Setiap image punya key tambahan 'path' (path absolut file gambar).
    """
    merge = merge or {}
    name_to_id = {n: i + 1 for i, n in enumerate(classes)}
    images, anns, dropped = [], [], Counter()
    next_img, next_ann = 1, 1
    for d in src_dirs:
        src = load_json(os.path.join(d, "_annotations.coco.json"))
        cat_name = {c["id"]: c["name"] for c in src["categories"]}
        by_img = defaultdict(list)
        for a in src["annotations"]:
            by_img[a["image_id"]].append(a)
        for im in src["images"]:
            path = os.path.join(d, im["file_name"])
            if not os.path.exists(path):
                dropped["<file gambar hilang>"] += 1
                continue
            with Image.open(path) as pil:
                w, h = pil.size
            scale = 1.0
            if max(w, h) > max_side:
                scale = target_side / max(w, h)
                nw, nh = round(w * scale), round(h * scale)
                os.makedirs(resized_dir, exist_ok=True)
                new_path = os.path.join(resized_dir, os.path.basename(d.rstrip("/\\")) + "_" + im["file_name"])
                if not os.path.exists(new_path):
                    with Image.open(path) as pil:
                        pil.convert("RGB").resize((nw, nh), Image.BILINEAR).save(new_path, quality=95)
                path = new_path
            else:
                nw, nh = w, h
            img_id = next_img
            next_img += 1
            images.append({"id": img_id, "file_name": os.path.basename(path), "width": nw, "height": nh,
                           "path": os.path.abspath(path), "orig_file_name": im["file_name"]})
            for a in by_img[im["id"]]:
                name = cat_name.get(a["category_id"], "?")
                name = merge.get(name, name)
                # "<kelas>-crowd" (mis. chair-crowd): satu polygon untuk kerumunan objek yang tidak bisa dipisah.
                # Disimpan sebagai <kelas> dengan iscrowd=1, sehingga pycocotools memperlakukannya sebagai
                # area abaikan (prediksi di dalamnya tidak dihitung salah, tidak terdeteksi juga tidak dihitung luput).
                crowd = isinstance(name, str) and name.endswith("-crowd")
                if crowd:
                    name = name[: -len("-crowd")]
                if name is None or name not in name_to_id:
                    if name is not None:
                        dropped[name] += 1
                    continue
                seg = a.get("segmentation")
                if isinstance(seg, dict):
                    if scale != 1.0:
                        seg = _resize_rle(seg, h, w, nh, nw)
                elif _poly_ok(seg):
                    seg = [[float(v) * scale for v in p] for p in seg if len(p) >= 6]
                else:
                    dropped["<tanpa polygon>"] += 1
                    continue
                x, y, bw, bh = a["bbox"]
                anns.append({"id": next_ann, "image_id": img_id, "category_id": name_to_id[name],
                             "segmentation": seg, "bbox": [x * scale, y * scale, bw * scale, bh * scale],
                             "area": float(a.get("area", bw * bh)) * scale * scale, "iscrowd": int(crowd)})
                next_ann += 1
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": i + 1, "name": n, "supercategory": "indoor"} for i, n in enumerate(classes)]}
    save_json(coco, out_json)
    if dropped:
        print(f"  [{os.path.basename(out_json)}] anotasi dibuang:", dict(dropped))
    return coco


# ----------------------------------------------------------------------------- subset bertingkat
def make_nested_subsets(train_coco, fractions, seed, out_dir):
    """Subset 25% ⊂ 50% ⊂ 100% dengan seed tetap.

    Urutan gambar: pertama, untuk setiap kelas (dari yang paling jarang) diambil satu gambar yang
    memuat kelas itu (supaya sebanyak mungkin kelas masih terwakili di subset terkecil), lalu sisa
    gambar diacak. Subset = prefix dari urutan ini, sehingga otomatis bertingkat.
    """
    rng = random.Random(seed)
    img_ids = sorted(im["id"] for im in train_coco["images"])
    img_cls, cls_imgs = defaultdict(set), defaultdict(set)
    for a in train_coco["annotations"]:
        img_cls[a["image_id"]].add(a["category_id"])
        cls_imgs[a["category_id"]].add(a["image_id"])
    order, chosen, covered = [], set(), set()
    for cid in sorted(cls_imgs, key=lambda c: (len(cls_imgs[c]), c)):
        if cid in covered:
            continue
        pick = rng.choice(sorted(cls_imgs[cid] - chosen))
        order.append(pick)
        chosen.add(pick)
        covered |= img_cls[pick]
    n_cover = len(order)
    rest = [i for i in img_ids if i not in chosen]
    rng.shuffle(rest)
    order += rest

    os.makedirs(out_dir, exist_ok=True)
    save_json({"seed": seed, "order": order, "n_coverage_images": n_cover}, os.path.join(out_dir, "train_order.json"))
    paths, rows = {}, defaultdict(dict)
    cat_names = {c["id"]: c["name"] for c in train_coco["categories"]}
    for f in sorted(fractions):
        n = max(1, round(f * len(img_ids)))
        keep = set(order[:n])
        sub = {"images": [im for im in train_coco["images"] if im["id"] in keep],
               "annotations": [a for a in train_coco["annotations"] if a["image_id"] in keep],
               "categories": train_coco["categories"]}
        p = os.path.join(out_dir, f"train_{frac_tag(f)}.json")
        save_json(sub, p)
        paths[f] = p
        cnt = Counter(a["category_id"] for a in sub["annotations"])
        for cid, name in cat_names.items():
            rows[name][f"inst_{frac_tag(f)}"] = cnt.get(cid, 0)
        n_cls = sum(1 for c in cat_names if cnt.get(c, 0) > 0)
        print(f"  subset {frac_tag(f)}: {len(sub['images'])} gambar, {len(sub['annotations'])} anotasi, "
              f"{n_cls}/{len(cat_names)} kelas punya contoh")
    with open(os.path.join(out_dir, "subset_class_counts.csv"), "w", newline="", encoding="utf-8") as fh:
        cols = [f"inst_{frac_tag(f)}" for f in sorted(fractions)]
        w = csv.writer(fh)
        w.writerow(["class"] + cols)
        for name in sorted(rows, key=lambda n: -rows[n][cols[-1]]):
            w.writerow([name] + [rows[name][c] for c in cols])
    if n_cover > round(min(fractions) * len(img_ids)):
        print(f"  PERINGATAN: butuh {n_cover} gambar untuk mencakup semua kelas, lebih dari subset terkecil.")
    return paths


def limit_coco(coco_path, n, out_path, seed=0):
    """Ambil n gambar pertama (untuk smoke test)."""
    c = load_json(coco_path)
    ids = sorted(im["id"] for im in c["images"])
    random.Random(seed).shuffle(ids)
    keep = set(ids[:n])
    c["images"] = [im for im in c["images"] if im["id"] in keep]
    c["annotations"] = [a for a in c["annotations"] if a["image_id"] in keep]
    save_json(c, out_path)
    return out_path


# ----------------------------------------------------------------------------- persiapan data lengkap
def prepare_data(cfg):
    """Download v5 (+ foto sendiri), seragamkan kelas, buat subset. Mengembalikan dict path."""
    data_dir = cfg["DATA_DIR"]
    canon = os.path.join(cfg["RESULTS_DIR"], "data_canonical")
    os.makedirs(canon, exist_ok=True)
    api_key = None

    v5 = cfg.get("V5_DIR")
    if not has_coco_split(v5):
        api_key = api_key or get_secret("ROBOFLOW_API_KEY")
        v5 = download_roboflow(api_key, "panopticindoor", "panoptic-indoor-segmentation", 5,
                               os.path.join(data_dir, "panoptic-indoor-v5-coco"))
    merge = cfg.get("CLASS_MERGE") or {}
    classes = canonical_classes(os.path.join(v5, "train", "_annotations.coco.json"), merge)
    print(f"Jumlah kelas: {len(classes)}")
    save_json(classes, os.path.join(canon, "classes.json"))

    out = {"classes": classes, "splits": {}}
    resized = os.path.join(data_dir, "resized")
    for split in ("train", "valid", "test"):
        name = "test_v5" if split == "test" else split
        p = os.path.join(canon, f"{name}.json")
        to_canonical_coco([os.path.join(v5, split)], classes, p, merge, resized, max_side=10 ** 9)  # v5 tidak diubah
        out["splits"][name] = p

    # Test set foto sendiri (opsional: dilewati kalau belum ada / gagal download)
    mc = cfg.get("MYCAPTURE") or {}
    if mc.get("ENABLED"):
        try:
            d = mc.get("DIR")
            if not has_coco_split(d):
                api_key = api_key or get_secret("ROBOFLOW_API_KEY")
                d = download_roboflow(api_key, mc["WORKSPACE"], mc["PROJECT"], mc["VERSION"],
                                      os.path.join(data_dir, "mycapture-coco"))
            split_dirs = [os.path.join(d, s) for s in ("train", "valid", "test")
                          if os.path.exists(os.path.join(d, s, "_annotations.coco.json"))]
            p = os.path.join(canon, "mycapture.json")
            c = to_canonical_coco(split_dirs, classes, p, merge, resized)
            print(f"  mycapture: {len(c['images'])} gambar, {len(c['annotations'])} anotasi")
            out["splits"]["mycapture"] = p
        except Exception as e:
            print("  PERINGATAN: test set foto sendiri dilewati ->", repr(e))

    if cfg.get("SMOKE"):
        for k in list(out["splits"]):
            n = 12 if k == "train" else 6
            out["splits"][k] = limit_coco(out["splits"][k], n, out["splits"][k].replace(".json", "_smoke.json"))

    train = load_json(out["splits"]["train"])
    for k, p in out["splits"].items():
        c = load_json(p)
        print(f"  {k:10s}: {len(c['images']):4d} gambar, {len(c['annotations']):5d} anotasi")
    out["subsets"] = make_nested_subsets(train, cfg["FRACTIONS"], cfg["SEED"], os.path.join(cfg["RESULTS_DIR"], "subsets"))
    return out


# ----------------------------------------------------------------------------- evaluasi
def mask_to_rle(binary):
    from pycocotools import mask as M
    rle = M.encode(np.asfortranarray(binary.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl"]


def coco_evaluate(gt_path, preds, max_det, iou_type="segm", img_ids=None):
    """mAP COCO via pycocotools. Return (dict ringkasan, dict AP per kelas).
    img_ids: kalau diisi, evaluasi hanya pada gambar-gambar itu (mis. per kondisi foto)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_path)
    names = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
    if not preds:
        return {k: 0.0 for k in STAT_NAMES}, {}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes([dict(p) for p in preds])
        E = COCOeval(gt, dt, iou_type)
        E.params.maxDets = [1, 10, max_det]
        if img_ids is not None:
            E.params.imgIds = sorted(img_ids)
        E.evaluate()
        E.accumulate()
    # Dihitung langsung dari tabel precision (E.stats bawaan pycocotools selalu memakai maxDets=100).
    prec = E.eval["precision"]  # [T(IoU), R(recall), K(kelas), A(area: all,s,m,l), M(maxDets)]
    ious = E.params.iouThrs

    def _ap(p):
        return float(np.mean(p[p > -1])) if (p > -1).any() else 0.0

    summary = {
        "AP": _ap(prec[:, :, :, 0, -1]),
        "AP50": _ap(prec[np.isclose(ious, 0.5)][:, :, :, 0, -1]),
        "AP75": _ap(prec[np.isclose(ious, 0.75)][:, :, :, 0, -1]),
        "APs": _ap(prec[:, :, :, 1, -1]),
        "APm": _ap(prec[:, :, :, 2, -1]),
        "APl": _ap(prec[:, :, :, 3, -1]),
    }
    per_class = {}
    for k_idx, cid in enumerate(E.params.catIds):
        p = prec[:, :, k_idx, 0, -1]
        if (p > -1).any():
            per_class[names[cid]] = float(np.mean(p[p > -1]))
    return summary, per_class


def evaluate_and_log(model_name, tag, split, gt_path, preds, run_dir, results_dir, max_det, extra=None):
    segm, per_class = coco_evaluate(gt_path, preds, max_det, "segm")
    bbox, _ = coco_evaluate(gt_path, preds, max_det, "bbox")
    row = {"model": model_name, "run": tag, "split": split, "n_pred": len(preds)}
    row.update({f"mask_{k}": round(v, 4) for k, v in segm.items()})
    row.update({f"box_{k}": round(v, 4) for k, v in bbox.items()})
    row.update(extra or {})
    append_csv(os.path.join(results_dir, "metrics.csv"), row)
    with open(os.path.join(run_dir, f"per_class_AP_{split}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "mask_AP"])
        for n, v in sorted(per_class.items(), key=lambda kv: -kv[1]):
            w.writerow([n, round(v, 4)])
    print(f"  {split:10s} mask AP={segm['AP']:.4f} AP50={segm['AP50']:.4f} | box AP={bbox['AP']:.4f}")
    return row


# ----------------------------------------------------------------------------- visualisasi
_PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                     [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
                     [0, 128, 128], [220, 190, 255], [170, 110, 40], [128, 0, 0], [170, 255, 195]], dtype=np.uint8)


def save_overlays(gt_path, preds, out_dir, classes, score_thr=0.3, max_images=60):
    """Simpan gambar dengan mask prediksi (skor >= score_thr) untuk analisis kegagalan."""
    from pycocotools import mask as M
    from PIL import ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    gt = load_json(gt_path)
    by_img = defaultdict(list)
    for p in preds:
        if p["score"] >= score_thr:
            by_img[p["image_id"]].append(p)
    for im in gt["images"][:max_images]:
        img = np.array(Image.open(im["path"]).convert("RGB")).astype(np.float32)
        labels = []
        for j, p in enumerate(sorted(by_img[im["id"]], key=lambda p: -p["score"])):
            m = M.decode(p["segmentation"]).astype(bool)
            if m.shape != img.shape[:2]:
                continue
            col = _PALETTE[(p["category_id"] - 1) % len(_PALETTE)]
            img[m] = img[m] * 0.5 + col * 0.5
            ys, xs = np.nonzero(m)
            if len(xs):
                labels.append((int(xs.min()), int(ys.min()), f"{classes[p['category_id'] - 1]} {p['score']:.2f}", col))
        pil = Image.fromarray(img.clip(0, 255).astype(np.uint8))
        dr = ImageDraw.Draw(pil)
        for x, y, t, col in labels:
            dr.rectangle([x, y, x + 7 * len(t), y + 12], fill=tuple(int(c) for c in col))
            dr.text((x + 1, y), t, fill=(0, 0, 0))
        pil.save(os.path.join(out_dir, os.path.splitext(im.get("orig_file_name", im["file_name"]))[0] + "_pred.jpg"), quality=90)


def run_predictions_and_eval(model_name, tag, predict_fn, data, cfg, run_dir, extra=None):
    """Prediksi + evaluasi di valid, test_v5, dan mycapture (kalau ada). predict_fn(coco_dict) -> list pred."""
    rows = []
    for split in ("valid", "test_v5", "mycapture"):
        if split not in data["splits"]:
            continue
        gt_path = data["splits"][split]
        t0 = time.time()
        preds = predict_fn(load_json(gt_path))
        if split != "valid":
            save_json(preds, os.path.join(run_dir, f"preds_{split}.json"))
        ex = dict(extra or {})
        ex["infer_sec"] = round(time.time() - t0, 1)
        rows.append(evaluate_and_log(model_name, tag, split, gt_path, preds, run_dir, cfg["RESULTS_DIR"],
                                     cfg["MAX_DET"], ex))
        if split == "mycapture":
            save_overlays(gt_path, preds, os.path.join(run_dir, "overlays_mycapture"), data["classes"])
    return rows


# ----------------------------------------------------------------------------- jadwal LR untuk model PyTorch
class TimeAwareCosine:
    """Cosine LR dengan warmup. Progress = max(iterasi/total, waktu/batas_waktu), jadi kalau batas
    waktu tercapai lebih dulu, jadwal LR tetap selesai turun (adil antar run)."""

    def __init__(self, optimizer, total_iters, warmup_iters, time_limit_sec):
        self.opt, self.total, self.warm, self.limit = optimizer, max(1, total_iters), warmup_iters, time_limit_sec
        self.base = [g["lr"] for g in optimizer.param_groups]
        self.t0, self.it = time.time(), 0

    def progress(self):
        p = self.it / self.total
        if self.limit:
            p = max(p, (time.time() - self.t0) / self.limit)
        return min(p, 1.0)

    def step(self):
        self.it += 1
        f = 0.5 * (1 + math.cos(math.pi * self.progress()))
        if self.it < self.warm:
            f *= self.it / self.warm
        for g, b in zip(self.opt.param_groups, self.base):
            g["lr"] = b * f

    def done(self):
        return self.progress() >= 1.0


def zip_results(results_dir):
    base = results_dir.rstrip("/\\")
    shutil.make_archive(base, "zip", results_dir)
    print("Hasil dikompres:", base + ".zip")
