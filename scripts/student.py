"""Distil the vision model into a tiny "student" that runs in milliseconds.

Teacher labels come from run_local.py debug files (full runs). The 14
hand-labelled test frames (eval/eval_images.json) are NEVER trained on:
they are only used to score the student with the official points.

  stock:      embedding (+ colour/texture stats) -> logistic regression ->
              P(empty/half/full) -> same expected-score rule as run_local.py
  compliance: k nearest training crops from OTHER planogram areas, labelled
              with the planogram product family; flag a mismatch only when
              most neighbours agree on a different family
  bare shelf: same colour/texture shortcut as run_local.py

    pip install scikit-learn torch transformers pillow          # once
    python scripts/student.py --teacher C:\\Deploy\\hackathon\\predictions.debug.json
    python scripts/student.py --teacher a.debug.json --teacher b.debug.json   # keep only agreed labels
    python scripts/student.py --embedder hist     # no download: colour histogram baseline
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
spec = importlib.util.spec_from_file_location("rl", ROOT / "run_local.py")
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)

LEVELS = ["empty", "half", "full"]
ORDER = {k: i for i, k in enumerate(LEVELS)}
# Products a ceiling camera can barely tell apart (same idea as the
# challenge's "recognised generically"). Keep in sync with run_local.py.
LOOKALIKE = [{"MAMAO", "PAPAIA"}, {"CITRINOS", "CLEMENTINA"}, {"LIMA", "LIMAO"},
             {"CEBOLA", "CEBOLINHA"}, {"ALPERCE", "PESSEGO", "NECTARINA"},
             {"MELANCIA", "MELOA"}]


def fam(name: str) -> str:
    f = rl.family(name)
    for g in LOOKALIKE:
        if f in g:
            return "|".join(sorted(g))
    return f


# ----------------------------------------------------------------- embedders
class HistEmbedder:
    """HSV histogram + texture stats. No download; a weak but instant baseline."""
    name = "hist"

    def __call__(self, crops):
        out = []
        for c in crops:
            hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 6, 6], [0, 180, 0, 256, 0, 256]).ravel()
            h = h / max(h.sum(), 1)
            out.append(np.sqrt(h))
        return np.asarray(out, dtype=np.float32)


class Dinov2Embedder:
    name = "dinov2"

    def __init__(self, model="facebook/dinov2-small"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval()

    def __call__(self, crops, bs=32):
        from PIL import Image
        feats = []
        for i in range(0, len(crops), bs):
            ims = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops[i:i + bs]]
            with self.torch.no_grad():
                o = self.model(**self.proc(images=ims, return_tensors="pt"))
            f = o.last_hidden_state[:, 0].numpy()
            feats.append(f / np.linalg.norm(f, axis=1, keepdims=True))
        return np.concatenate(feats).astype(np.float32)


# ------------------------------------------------------------------- data
def teacher_labels(paths, planogram):
    """{(cam,img,area): level} from one or more run_local debug files (agreeing only)."""
    per = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        lab = {}
        for cam, imgs in d.items():
            exp = {a["area_id"]: a["name"] for a in planogram[cam]}
            for img, areas in imgs.items():
                for aid, info in areas.items():
                    if info.get("colour_override"):
                        lab[(cam, img, aid)] = "empty"
                    elif info.get("samples"):
                        res, _ = rl.decide(info["samples"], exp[aid], 0.0)
                        lab[(cam, img, aid)] = res["stock_level"]
        per.append(lab)
    keys = set(per[0]).intersection(*per[1:]) if len(per) > 1 else set(per[0])
    out = {k: per[0][k] for k in keys if all(p[k] == per[0][k] for p in per)}
    return out, (len(keys) - len(out))


def crops_for(keys, images_dir, planogram):
    by_img = {}
    for cam, img, aid in keys:
        by_img.setdefault((cam, img), []).append(aid)
    crops, order, extra = [], [], []
    for (cam, img), aids in sorted(by_img.items()):
        im = cv2.imread(str(Path(images_dir) / cam / img))
        if im is None:
            continue
        quads = {a["area_id"]: a["quad"] for a in planogram[cam]}
        for aid in aids:
            c = rl.warp_area(im, quads[aid], (224, 224), 0.08)
            f = rl.shelf_features(c)
            crops.append(c); order.append((cam, img, aid))
            extra.append([f["dark"], f["fruit"], f["texture"] * 10, f["v_std"] / 64])
    return crops, order, np.asarray(extra, dtype=np.float32)


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", action="append", required=True)
    ap.add_argument("--embedder", default="dinov2", choices=["dinov2", "hist"])
    ap.add_argument("--images-dir", default=str(ROOT / "data" / "Frutas e Legumes (train & test)"))
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--mismatch-share", type=float, default=0.7,
                    help="flag non-compliant only if this share of neighbours is another family")
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression

    pg = rl.load_planogram(str(ROOT / "planogram.json"))
    expected = {a_["area_id"]: a_["name"] for v in pg.values() for a_ in v}
    test_imgs = json.loads((EVAL / "eval_images.json").read_text(encoding="utf-8"))
    labels = json.loads((EVAL / "labels.json").read_text(encoding="utf-8"))

    lab, disagree = teacher_labels(a.teacher, pg)
    lab = {k: v for k, v in lab.items() if test_imgs.get(k[0]) != k[1]}   # hold out test frames
    print(f"teacher labels: {len(lab)} areas (dropped {disagree} where teachers disagree)",
          dict(Counter(lab.values())))

    emb = HistEmbedder() if a.embedder == "hist" else Dinov2Embedder()
    tr_crops, tr_keys, tr_extra = crops_for(lab.keys(), a.images_dir, pg)
    t0 = time.time()
    Xtr = np.hstack([emb(tr_crops), tr_extra])
    print(f"embedded {len(tr_crops)} training crops in {time.time() - t0:.1f}s")
    ytr = np.array([ORDER[lab[k]] for k in tr_keys])

    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xtr, ytr)

    # neighbours for compliance: only stocked training crops, labelled by planogram family
    stocked = ytr > 0
    nb_X, nb_area = Xtr[stocked], [tr_keys[i][2] for i in np.where(stocked)[0]]
    nb_fam = [fam(expected[x]) for x in nb_area]
    D = Xtr.shape[1] - tr_extra.shape[1]

    te_keys_all = [(cam, img, aid) for cam, img in test_imgs.items() for aid in labels[cam][img]]
    t0 = time.time()
    te_crops, te_keys, te_extra = crops_for(te_keys_all, a.images_dir, pg)
    Xte = np.hstack([emb(te_crops), te_extra])
    P = clf.predict_proba(Xte)
    pts = ok_id = ok_st = below = above = 0
    for i, (cam, img, aid) in enumerate(te_keys):
        f = rl.shelf_features(cv2.resize(te_crops[i], (512, 384)))
        if f["dark"] > 0.60 and f["fruit"] < 0.06 and f["texture"] < 0.04:
            lvl, comp = "empty", True
        else:
            lvl = LEVELS[rl.best_level(list(P[i]))]
            comp = True
            if lvl != "empty" and len(nb_X):
                sims = nb_X[:, :D] @ Xte[i, :D]
                idx = [j for j in np.argsort(-sims) if nb_area[j] != aid][: a.k]
                top, n = Counter(nb_fam[j] for j in idx).most_common(1)[0]
                if top != fam(expected[aid]) and n / len(idx) >= a.mismatch_share:
                    comp = False
        t = labels[cam][img][aid]
        if comp == t["compliant"]:
            pts += 1; ok_id += 1
        else:
            pts -= 2
        pl, tl = ORDER[lvl], ORDER[t["stock"]]
        if pl == tl:
            pts += 1; ok_st += 1
        elif pl < tl:
            pts -= 2; below += 1
        else:
            pts -= 3; above += 1
    dt = time.time() - t0
    n = len(te_keys)
    print(f"\nSTUDENT ({emb.name}) on the {n} hand-labelled test areas:")
    print(f"  points {pts}/{2 * n}   id ok {ok_id}   stock ok {ok_st}   below {below}   above {above}")
    print(f"  {dt:.1f}s for {len(test_imgs)} images = {dt / len(test_imgs):.2f}s per image (incl. embedding, CPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
