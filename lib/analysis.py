"""Helper analisis untuk notebook laporan: grafik, statistik per foto, visualisasi studi kasus."""
import csv
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as M

# Warna per model (palet kategorikal tervalidasi, urutan tetap) + marker berbeda sebagai encoding kedua
COLORS = {"yolo11s": "#2a78d6", "maskrcnn": "#eb6834", "mask2former": "#1baf7a"}
MARKERS = {"yolo11s": "o", "maskrcnn": "s", "mask2former": "^"}
LABELS = {"yolo11s": "YOLO11s-seg", "maskrcnn": "Mask R-CNN", "mask2former": "Mask2Former"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e0", "#fcfcfb"
FRACS = ["025pct", "050pct", "100pct"]
FRAC_LABEL = {"025pct": "25%", "050pct": "50%", "100pct": "100%"}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False, "figure.dpi": 110,
    })


# ----------------------------------------------------------------------------- grafik
def plot_fraction_lines(df, splits, value="mask_AP", titles=None):
    """Small multiples: AP vs jumlah data latih, satu panel per split, satu garis per model (skala y sama)."""
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(FRACS))
    for ax, split in zip(axes, splits):
        for key in COLORS:
            d = df[(df.model_key == key) & (df.split == split)].set_index("run").reindex(FRACS)
            ax.plot(x, d[value], color=COLORS[key], marker=MARKERS[key], lw=2, ms=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[key])
            last = d[value].iloc[-1]
            if pd.notna(last):
                ax.annotate(f"{last:.3f}", (x[-1], last), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=9, color=INK2)
        ax.set_xticks(x, [FRAC_LABEL[f] for f in FRACS])
        ax.set_xlim(-0.25, len(FRACS) - 0.4)
        ax.set_title((titles or {}).get(split, split))
        ax.set_xlabel("data latih")
    axes[0].set_ylabel(value.replace("_", " "))
    axes[0].set_ylim(bottom=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=9, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    return fig


def plot_domain_shift(df, run="100pct", value="mask_AP"):
    """Slope chart: test v5 -> mycapture untuk tiap model."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for key in COLORS:
        d = df[(df.model_key == key) & (df.run == run)].set_index("split")
        ys = [d.loc["test_v5", value], d.loc["mycapture", value]]
        ax.plot([0, 1], ys, color=COLORS[key], marker=MARKERS[key], lw=2, ms=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[key])
        ax.annotate(f"{LABELS[key]}  {ys[1]:.3f}", (1, ys[1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
        ax.annotate(f"{ys[0]:.3f}", (0, ys[0]), xytext=(-8, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=9, color=INK2)
    ax.set_xticks([0, 1], ["test v5\n(asli)", "mycapture\n(foto sendiri)"])
    ax.set_xlim(-0.45, 1.9)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(value.replace("_", " "))
    ax.set_title(f"Domain shift (run {FRAC_LABEL[run]})")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- kondisi foto
def photo_number(file_name):
    m = re.search(r"mycapture_(\d+)", file_name)
    return int(m.group(1)) if m else None


def load_conditions(rename_log, coco):
    """image_id -> set label kondisi (dari kolom 'kondisi' di rename_log.csv)."""
    by_num = {}
    with open(rename_log, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_num[int(r["index"])] = set(filter(None, r.get("kondisi", "").split(";")))
    out = {}
    for im in coco["images"]:
        n = photo_number(im.get("orig_file_name", im["file_name"]))
        out[im["id"]] = by_num.get(n, set())
    return out


# ----------------------------------------------------------------------------- statistik per foto
def _rles(objs, h, w):
    out = []
    for o in objs:
        s = o["segmentation"]
        if isinstance(s, list):
            out.append(M.merge(M.frPyObjects(s, h, w)))
        elif isinstance(s.get("counts"), list):
            out.append(M.frPyObjects(s, h, w))
        else:
            out.append(s)
    return out


def image_stats(gt, preds, iou_thr=0.5, score_thr=0.3):
    """Per foto: berapa objek GT terdeteksi benar (IoU >= iou_thr, kelas sama), luput, prediksi salah,
    dan prediksi yang menutupi objek dengan kelas lain (salah kelas). Crowd diabaikan."""
    by_gt, by_pr = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        by_gt[a["image_id"]].append(a)
    for p in preds:
        if p["score"] >= score_thr:
            by_pr[p["image_id"]].append(p)
    rows = []
    for im in gt["images"]:
        g = [a for a in by_gt[im["id"]] if not a.get("iscrowd")]
        crowd = [a for a in by_gt[im["id"]] if a.get("iscrowd")]
        p = sorted(by_pr[im["id"]], key=lambda d: -d["score"])
        h, w = im["height"], im["width"]
        tp = wrong_cls = fp = 0
        if g and p:
            iou = M.iou(_rles(p, h, w), _rles(g, h, w), [0] * len(g))
            used = set()
            for i, d in enumerate(p):
                same = [j for j in range(len(g)) if g[j]["category_id"] == d["category_id"] and j not in used]
                j = max(same, key=lambda j: iou[i, j], default=None)
                if j is not None and iou[i, j] >= iou_thr:
                    used.add(j)
                    tp += 1
                elif iou[i].max() >= iou_thr:
                    wrong_cls += 1
                else:
                    fp += 1
        else:
            fp = len(p)
        if crowd and fp:   # prediksi di dalam area crowd tidak dihitung salah
            fp = max(0, fp - _inside_crowd(p, crowd, h, w))
        rows.append({"image_id": im["id"], "foto": photo_number(im.get("orig_file_name", im["file_name"])),
                     "n_gt": len(g), "benar": tp, "luput": len(g) - tp, "salah_kelas": wrong_cls,
                     "prediksi_salah": fp, "recall": tp / len(g) if g else np.nan})
    return pd.DataFrame(rows)


def _inside_crowd(p, crowd, h, w):
    if not p:
        return 0
    iou = M.iou(_rles(p, h, w), _rles(crowd, h, w), [1] * len(crowd))
    return int((iou.max(1) >= 0.5).sum())


# ----------------------------------------------------------------------------- visualisasi studi kasus
_PAL = np.array([[230, 25, 75], [60, 180, 75], [255, 200, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
                 [70, 220, 220], [240, 50, 230], [160, 200, 40], [0, 128, 128], [170, 110, 40], [128, 0, 0]],
                dtype=np.float32)


def overlay(img, objs, classes, h, w, score_thr=0.3, only=None, label=True, ax=None):
    """Gambar mask (GT atau prediksi) di atas foto. only: set category_id yang ditampilkan."""
    arr = np.array(img).astype(np.float32)
    objs = [o for o in objs if o.get("score", 1) >= score_thr and (only is None or o["category_id"] in only)]
    texts = []
    for o, r in zip(objs, _rles(objs, h, w)):
        m = M.decode(r).astype(bool)
        col = _PAL[(o["category_id"] * 7) % len(_PAL)]
        arr[m] = arr[m] * 0.45 + col * 0.55
        edge = m ^ np.roll(m, 1, 0) | m ^ np.roll(m, 1, 1)
        arr[edge] = col * 0.6
        if label and m.any():
            ys, xs = np.nonzero(m)
            name = classes[o["category_id"] - 1] + ("*" if o.get("iscrowd") else "")
            txt = name if "score" not in o else f"{name} {o['score']:.2f}"
            texts.append((xs.mean(), ys.mean(), txt))
    ax = ax or plt.gca()
    ax.imshow(arr.clip(0, 255).astype(np.uint8))
    for x, y, t in texts[:40]:
        ax.text(x, y, t, fontsize=6.5, color="white", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=(0, 0, 0, 0.55), ec="none"))
    ax.set_axis_off()
    return ax


def show_case(image_id, gt, preds_by_model, classes, score_thr=0.3, only=None, figsize=(15, 3.6)):
    """Satu baris: foto + anotasi (GT) | prediksi tiap model."""
    im = next(i for i in gt["images"] if i["id"] == image_id)
    img = Image.open(im["path"]).convert("RGB")
    h, w = im["height"], im["width"]
    g = [a for a in gt["annotations"] if a["image_id"] == image_id]
    n = 1 + len(preds_by_model)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    overlay(img, g, classes, h, w, only=only, ax=axes[0])
    axes[0].set_title("Anotasi (GT)", fontsize=10)
    for ax, (key, preds) in zip(axes[1:], preds_by_model.items()):
        p = [d for d in preds if d["image_id"] == image_id]
        overlay(img, p, classes, h, w, score_thr=score_thr, only=only, ax=ax)
        ax.set_title(f"{LABELS.get(key, key)} (skor ≥ {score_thr})", fontsize=10)
    fig.tight_layout()
    return fig


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else np.nan


# ----------------------------------------------------------------------------- helper eksperimen
def roi_mask_upper_bound(gt, res=28, classes=None):
    """Batas atas kualitas mask kalau mask hanya disimpan di grid res×res di dalam box (seperti mask head Mask R-CNN).
    Untuk setiap objek GT: potong mask sesuai box, kecilkan ke res×res, besarkan lagi ke ukuran box, threshold 0,5,
    lalu hitung IoU dengan mask asli. IoU < 1 berarti detail hilang karena resolusi grid, bukan karena model salah."""
    rows = []
    ims = {i["id"]: i for i in gt["images"]}
    for a in gt["annotations"]:
        if a.get("iscrowd"):
            continue
        im = ims[a["image_id"]]
        m = M.decode(_rles([a], im["height"], im["width"])[0]).astype(np.float32)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        crop = m[y0:y1, x0:x1]
        small = np.array(Image.fromarray(crop).resize((res, res), Image.BILINEAR))
        back = np.array(Image.fromarray(small).resize((x1 - x0, y1 - y0), Image.BILINEAR)) > 0.5
        rows.append({"image_id": a["image_id"], "foto": photo_number(im.get("orig_file_name", im["file_name"])),
                     "kelas": classes[a["category_id"] - 1] if classes else a["category_id"],
                     "area": int(crop.sum()), "box_w": int(x1 - x0), "box_h": int(y1 - y0),
                     "iou_maks": mask_iou(crop > 0.5, back)})
    return pd.DataFrame(rows)


def gamma_correct(img, gamma=0.5):
    """gamma < 1 mencerahkan bagian gelap (untuk foto redup/backlight)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    return Image.fromarray((arr ** gamma * 255).clip(0, 255).astype(np.uint8))


def make_variant(gt, img_ids, fn, out_dir, out_json):
    """Buat salinan subset test set dengan gambar yang diubah oleh fn(PIL) -> PIL (anotasi tetap)."""
    import os
    from common import save_json
    os.makedirs(out_dir, exist_ok=True)
    ids = set(img_ids)
    images = []
    for im in gt["images"]:
        if im["id"] not in ids:
            continue
        p = os.path.join(out_dir, os.path.basename(im["path"]))
        fn(Image.open(im["path"]).convert("RGB")).save(p, quality=95)
        images.append(dict(im, path=os.path.abspath(p)))
    c = {"images": images, "annotations": [a for a in gt["annotations"] if a["image_id"] in ids],
         "categories": gt["categories"]}
    save_json(c, out_json)
    return c


def matched_iou(gt, preds, image_id, category_id, score_thr=0.3):
    """Untuk setiap objek GT kelas tertentu di satu foto: IoU tertinggi dengan prediksi kelas yang sama."""
    im = next(i for i in gt["images"] if i["id"] == image_id)
    h, w = im["height"], im["width"]
    g = [a for a in gt["annotations"] if a["image_id"] == image_id and a["category_id"] == category_id and not a.get("iscrowd")]
    p = [d for d in preds if d["image_id"] == image_id and d["category_id"] == category_id and d["score"] >= score_thr]
    if not g:
        return np.array([])
    if not p:
        return np.zeros(len(g))
    return M.iou(_rles(p, h, w), _rles(g, h, w), [0] * len(g)).max(0)


def crop_zoom(gt, image_id, box, out_dir, out_json, long_side=640):
    """Potong satu foto ke box (x0, y0, x1, y1), perbesar sampai sisi panjang = long_side, anotasi ikut dipotong.
    Dipakai untuk menguji apakah objek gagal dideteksi karena terlalu kecil di input 640."""
    import os
    from common import save_json, mask_to_rle
    os.makedirs(out_dir, exist_ok=True)
    im = next(i for i in gt["images"] if i["id"] == image_id)
    x0, y0, x1, y1 = [int(v) for v in box]
    s = long_side / max(x1 - x0, y1 - y0)
    nw, nh = round((x1 - x0) * s), round((y1 - y0) * s)
    img = Image.open(im["path"]).convert("RGB").crop((x0, y0, x1, y1)).resize((nw, nh), Image.BICUBIC)
    p = os.path.abspath(os.path.join(out_dir, f"zoom_{image_id}.jpg"))
    img.save(p, quality=95)
    anns = []
    for a in gt["annotations"]:
        if a["image_id"] != image_id:
            continue
        m = M.decode(_rles([a], im["height"], im["width"])[0])[y0:y1, x0:x1]
        if m.sum() < 4:
            continue
        m = np.array(Image.fromarray(m * 255).resize((nw, nh), Image.NEAREST)) > 127
        ys, xs = np.nonzero(m)
        anns.append(dict(a, id=len(anns) + 1, segmentation=mask_to_rle(m), area=float(m.sum()),
                         bbox=[float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]))
    c = {"images": [dict(im, path=p, width=nw, height=nh)], "annotations": anns, "categories": gt["categories"]}
    save_json(c, out_json)
    return c


def union_box(gt, image_id, category_ids, pad=10):
    """Box yang mencakup semua objek GT dengan kelas tertentu di satu foto."""
    im = next(i for i in gt["images"] if i["id"] == image_id)
    bs = [a["bbox"] for a in gt["annotations"] if a["image_id"] == image_id and a["category_id"] in category_ids]
    x0 = max(0, min(b[0] for b in bs) - pad); y0 = max(0, min(b[1] for b in bs) - pad)
    x1 = min(im["width"], max(b[0] + b[2] for b in bs) + pad); y1 = min(im["height"], max(b[1] + b[3] for b in bs) + pad)
    return x0, y0, x1, y1


def predicted_at(gt, preds, image_id, category_id, classes, score_thr=0.1, iou_thr=0.3):
    """Untuk setiap objek GT kelas tertentu: kelas apa yang diprediksi model di lokasi itu (IoU >= iou_thr).
    '(tidak terdeteksi)' kalau tidak ada prediksi yang cukup menutupi objek."""
    from collections import Counter
    im = next(i for i in gt["images"] if i["id"] == image_id)
    h, w = im["height"], im["width"]
    g = [a for a in gt["annotations"] if a["image_id"] == image_id and a["category_id"] == category_id and not a.get("iscrowd")]
    p = [d for d in preds if d["image_id"] == image_id and d["score"] >= score_thr]
    if not g:
        return Counter()
    if not p:
        return Counter({"(tidak terdeteksi)": len(g)})
    iou = M.iou(_rles(p, h, w), _rles(g, h, w), [0] * len(g))
    out = Counter()
    for j in range(len(g)):
        i = int(iou[:, j].argmax())
        out[classes[p[i]["category_id"] - 1] if iou[i, j] >= iou_thr else "(tidak terdeteksi)"] += 1
    return out
