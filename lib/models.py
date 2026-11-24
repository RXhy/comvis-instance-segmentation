"""Memuat bobot hasil training (Kaggle) dan menjalankan inferensi untuk ketiga model.

Preprocessing dan post-processing di sini SAMA dengan notebook training 02-04, supaya angka AP yang dihitung
ulang di laptop bisa dibandingkan dengan angka dari Kaggle.

Pemakaian:
    m = load_model("mask2former", "mask2former/100pct", classes)
    hasil = m.predict_image("foto.jpg")                 # list of dict(mask=bool HxW, category_id, score)
    preds = m.predict_coco(coco_dict)                   # format hasil COCO (RLE) untuk pycocotools
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from common import mask_to_rle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

MODEL_INFO = {
    "yolo11s": dict(label="YOLO11s-seg", family="one-stage CNN", folder="yolo11s"),
    "maskrcnn": dict(label="Mask R-CNN R50-FPN v2", family="two-stage CNN", folder="maskrcnn"),
    "mask2former": dict(label="Mask2Former Swin-T", family="transformer", folder="mask2former"),
}


def _to_pil(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    # exif_transpose: foto HP langsung (mis. saat demo) diputar sesuai EXIF; export Roboflow sudah tegak
    return ImageOps.exif_transpose(Image.open(img)).convert("RGB")


class _Base:
    key = ""

    def __init__(self, classes, score_thresh=0.05, max_det=200):
        self.classes = classes
        self.score_thresh = score_thresh
        self.max_det = max_det

    def predict_image(self, img, **kw):
        raise NotImplementedError

    def predict_coco(self, coco, **kw):
        """Prediksi untuk semua gambar di dict COCO (butuh key 'path'), hasil dalam format COCO results."""
        preds = []
        for im in coco["images"]:
            for d in self.predict_image(im["path"], **kw):
                m = d["mask"]
                if m.shape != (im["height"], im["width"]):
                    m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
                        (im["width"], im["height"]), Image.NEAREST)) > 127
                if m.any():
                    preds.append({"image_id": im["id"], "category_id": d["category_id"],
                                  "segmentation": mask_to_rle(m), "score": d["score"]})
        return preds


# ----------------------------------------------------------------------------- YOLO11s-seg
class YoloSeg(_Base):
    key = "yolo11s"

    def __init__(self, run_dir, classes, **kw):
        super().__init__(classes, **kw)
        from ultralytics import YOLO
        self.model = YOLO(os.path.join(run_dir, "weights", "best.pt"))

    def predict_image(self, img, imgsz=640, retina_masks=True, score_thresh=None, iou=0.7):
        """imgsz dan retina_masks bisa diubah untuk eksperimen resolusi prototype mask."""
        pil = _to_pil(img)
        r = self.model.predict(pil, imgsz=imgsz, conf=score_thresh or self.score_thresh, iou=iou,
                               max_det=self.max_det, retina_masks=retina_masks, verbose=False,
                               device=0 if DEVICE == "cuda" else "cpu")[0]
        if r.masks is None:
            return []
        masks = r.masks.data.cpu().numpy() > 0.5
        return [{"mask": m, "category_id": int(c) + 1, "score": float(s)}
                for m, c, s in zip(masks, r.boxes.cls.tolist(), r.boxes.conf.tolist())]


# ----------------------------------------------------------------------------- Mask R-CNN
class MaskRCNN(_Base):
    key = "maskrcnn"
    MIN_SIZE, MAX_SIZE = 480, 640

    def __init__(self, run_dir, classes, **kw):
        super().__init__(classes, **kw)
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        # weights=None: arsitektur saja, bobotnya diisi dari best.pt hasil training
        m = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=self.MIN_SIZE,
                                     max_size=self.MAX_SIZE, box_detections_per_img=self.max_det,
                                     box_score_thresh=self.score_thresh)
        k = len(classes) + 1   # + background
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, k)
        m.roi_heads.mask_predictor = MaskRCNNPredictor(m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, k)
        m.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=True))
        self.model = m.to(DEVICE).eval()

    @torch.no_grad()
    def predict_image(self, img, score_thresh=None):
        pil = _to_pil(img)
        x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float().div(255).to(DEVICE)
        self.model.roi_heads.score_thresh = score_thresh or self.score_thresh
        with torch.autocast("cuda", enabled=USE_AMP):
            out = self.model([x])[0]
        masks = (out["masks"][:, 0] > 0.5).cpu().numpy()
        return [{"mask": m, "category_id": int(c), "score": float(s)}
                for m, c, s in zip(masks, out["labels"].tolist(), out["scores"].tolist())]


# ----------------------------------------------------------------------------- Mask2Former
class Mask2Former(_Base):
    key = "mask2former"
    CKPT = "facebook/mask2former-swin-tiny-coco-instance"
    LONG_SIDE = 640

    def __init__(self, run_dir, classes, **kw):
        super().__init__(classes, **kw)
        from transformers import AutoImageProcessor, Mask2FormerConfig, Mask2FormerForUniversalSegmentation
        cfg = Mask2FormerConfig.from_pretrained(self.CKPT, id2label={i: n for i, n in enumerate(classes)},
                                                label2id={n: i for i, n in enumerate(classes)})
        m = Mask2FormerForUniversalSegmentation(cfg)
        m.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=True))
        self.model = m.to(DEVICE).eval()
        proc = AutoImageProcessor.from_pretrained(self.CKPT)
        self.mean = np.array(proc.image_mean, dtype=np.float32)
        self.std = np.array(proc.image_std, dtype=np.float32)
        if self.mean.max() > 1:
            self.mean, self.std = self.mean / 255, self.std / 255

    @torch.no_grad()
    def predict_image(self, img, long_side=None, score_thresh=None):
        """long_side bisa diubah untuk eksperimen resolusi input."""
        pil = _to_pil(img)
        w, h = pil.size
        s = (long_side or self.LONG_SIDE) / max(w, h)
        nw, nh = round(w * s), round(h * s)
        arr = (np.asarray(pil.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255 - self.mean) / self.std
        x = torch.from_numpy(arr).permute(2, 0, 1)
        H, W = (nh + 31) // 32 * 32, (nw + 31) // 32 * 32          # padding ke kelipatan 32, seperti saat training
        pv = torch.zeros(1, 3, H, W)
        pm = torch.zeros(1, H, W, dtype=torch.long)
        pv[0, :, :nh, :nw] = x
        pm[0, :nh, :nw] = 1
        with torch.autocast("cuda", enabled=USE_AMP):
            out = self.model(pixel_values=pv.to(DEVICE), pixel_mask=pm.to(DEVICE))
        cls = out.class_queries_logits[0].float().softmax(-1)[:, :-1]          # (Q, K), kolom terakhir = "no object"
        Q, K = cls.shape
        scores, idx = cls.flatten().topk(min(self.max_det, Q * K))
        q, lab = idx // K, idx % K
        ml = out.masks_queries_logits[0].float()[q]                             # (D, H/4, W/4)
        ml = F.interpolate(ml[None], size=(H, W), mode="bilinear", align_corners=False)[0][:, :nh, :nw]
        thr = score_thresh or self.score_thresh
        res = []
        chunk = max(1, min(50, int(4e7 // (h * w))))
        for j in range(0, len(ml), chunk):
            prob = F.interpolate(ml[None, j: j + chunk], size=(h, w), mode="bilinear", align_corners=False)[0].sigmoid()
            binary = prob > 0.5
            area = binary.flatten(1).sum(1, dtype=torch.float32)
            mscore = (prob * binary).flatten(1).sum(1) / (area + 1e-6)
            final = scores[j: j + chunk] * mscore
            for k in range(len(prob)):
                if area[k] > 0 and final[k] >= thr:
                    res.append({"mask": binary[k].cpu().numpy(), "category_id": int(lab[j + k]) + 1,
                                "score": float(final[k])})
        return res


_CLASSES = {"yolo11s": YoloSeg, "maskrcnn": MaskRCNN, "mask2former": Mask2Former}


def load_model(key, run_dir, classes, **kw):
    return _CLASSES[key](run_dir, classes, **kw)


def free_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
