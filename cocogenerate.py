import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import fiftyone as fo
import fiftyone.zoo as foz


# ----------------------------
# Image transforms (tile-level)
# ----------------------------
def to_grayscale_bgr(img_bgr):
    """Return a 3-channel BGR grayscale image (keeps panel consistent)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def blur(img_bgr, k=5):
    return cv2.blur(img_bgr, (k, k))

def low_res(img_bgr, downscale=4):
    """
    Simulate low resolution by downscaling then upscaling.
    downscale=4 means shrink to 1/4 width/height then resize back.
    """
    h, w = img_bgr.shape[:2]
    small_w = max(1, w // downscale)
    small_h = max(1, h // downscale)
    small = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def low_saturation(img_bgr, sat_scale=0.3):
    """
    Reduce saturation in HSV space. sat_scale in [0,1].
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[..., 1] = hsv[..., 1] * sat_scale
    hsv[..., 1] = hsv[..., 1].clip(0, 255)
    hsv = hsv.astype("uint8")
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_transform(img_bgr, transform_name: str):
    if transform_name == "none":
        return img_bgr
    if transform_name == "grayscale":
        return to_grayscale_bgr(img_bgr)
    if transform_name == "blur":
        return blur(img_bgr, k=5)
    if transform_name == "grayscale_blur":
        return blur(to_grayscale_bgr(img_bgr), k=5)
    if transform_name == "low_res":
        return low_res(img_bgr, downscale=4)
    if transform_name == "low_sat":
        return low_saturation(img_bgr, sat_scale=0.3)
    if transform_name == "low_res_low_sat":
        return low_saturation(low_res(img_bgr, downscale=4), sat_scale=0.3)

    raise ValueError(f"Unknown transform: {transform_name}")


# -----------------------------------------
# Bounding box conversion / panel remapping
# -----------------------------------------
def rel_xywh_to_abs_xywh(rel_xywh, img_w: int, img_h: int):
    """
    FiftyOne detections typically store bounding_box as [x, y, w, h] normalized to [0,1]
    with x,y being top-left.
    """
    x_rel, y_rel, w_rel, h_rel = rel_xywh
    x = x_rel * img_w
    y = y_rel * img_h
    w = w_rel * img_w
    h = h_rel * img_h
    return [x, y, w, h]

def scale_xywh(xywh, sx: float, sy: float):
    x, y, w, h = xywh
    return [x * sx, y * sy, w * sx, h * sy]

def clamp_xywh(xywh, max_w: int, max_h: int):
    x, y, w, h = xywh
    x = max(0.0, min(x, float(max_w)))
    y = max(0.0, min(y, float(max_h)))
    w = max(0.0, min(w, float(max_w) - x))
    h = max(0.0, min(h, float(max_h) - y))
    return [x, y, w, h]


# ----------------------------
# Panel generation
# ----------------------------
def build_panel_2x2(
    imgs_bgr: List,
    tile_size: int,
) -> Tuple:
    """
    Resize each image to (tile_size, tile_size) and stitch into 2x2.
    Returns: (panel_img, offsets), where offsets are (x_off, y_off) per quadrant.
    """
    assert len(imgs_bgr) == 4

    tiles = []
    for img in imgs_bgr:
        tile = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        tiles.append(tile)

    top = cv2.hconcat([tiles[0], tiles[1]])
    bottom = cv2.hconcat([tiles[2], tiles[3]])
    panel = cv2.vconcat([top, bottom])

    # Quadrant offsets in panel coordinates
    offsets = [
        (0, 0),                 # top-left
        (tile_size, 0),          # top-right
        (0, tile_size),          # bottom-left
        (tile_size, tile_size),  # bottom-right
    ]

    return panel, offsets


def main(
    output_dir: str,
    num_panels: int = 1000,
    split: str = "validation",
    max_source_images: int = 4000,
    tile_size: int = 512,
    seed: int = 7,
    transform_policy: str = "random",
):
    """
    transform_policy:
      - "none": no transform
      - "random": randomly assign a transform per tile from a list
      - "lowres_vs_normal": half tiles low_res, half none (simple controlled test)
    """
    random.seed(seed)
    out = Path(output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    # Download/load COCO via FiftyOne
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"],  # gives sample.ground_truth
        max_samples=max_source_images,
    )

    # Snapshot samples into list for quick random access
    samples = list(dataset)
    if len(samples) < 4:
        raise RuntimeError("Not enough samples loaded to build panels.")

    # Define transforms you want to study
    transform_choices = [
        "none",
        "low_res",
        "low_sat",
        "low_res_low_sat",
        "grayscale",
        "blur",
        "grayscale_blur",
    ]

    for panel_idx in range(num_panels):
        chosen = random.sample(samples, 4)

        # Load + optionally transform per tile
        imgs = []
        tile_transforms = []
        source_meta = []

        for s_i, s in enumerate(chosen):
            img = cv2.imread(s.filepath)
            if img is None:
                raise RuntimeError(f"Failed to read image: {s.filepath}")

            if transform_policy == "none":
                tname = "none"
            elif transform_policy == "lowres_vs_normal":
                tname = "low_res" if (s_i % 2 == 0) else "none"
            elif transform_policy == "random":
                tname = random.choice(transform_choices)
            else:
                raise ValueError(f"Unknown transform_policy: {transform_policy}")

            img_t = apply_transform(img, tname)

            imgs.append(img_t)
            tile_transforms.append(tname)

            # Best-effort IDs: COCO sample has fields, but not guaranteed in all FO versions.
            source_meta.append({
                "filepath": s.filepath,
                "sample_id": str(s.id),
            })

        # Stitch into 2x2 panel
        panel_img, offsets = build_panel_2x2(imgs, tile_size=tile_size)

        panel_h, panel_w = panel_img.shape[:2]

        # Merge detections with remapped bboxes
        merged_dets = []

        for q, s in enumerate(chosen):
            x_off, y_off = offsets[q]

            # Read original dims (needed for bbox conversion)
            orig = cv2.imread(s.filepath)
            if orig is None:
                raise RuntimeError(f"Failed to read image for bbox remap: {s.filepath}")
            orig_h, orig_w = orig.shape[:2]

            sx = tile_size / float(orig_w)
            sy = tile_size / float(orig_h)

            # FiftyOne COCO detections live at sample.ground_truth.detections
            gt = getattr(s, "ground_truth", None)
            if gt is None or gt.detections is None:
                continue

            for det in gt.detections:
                # det.bounding_box is normalized [x,y,w,h]
                abs_xywh = rel_xywh_to_abs_xywh(det.bounding_box, orig_w, orig_h)
                scaled_xywh = scale_xywh(abs_xywh, sx, sy)

                # Shift into panel coords
                x, y, w, h = scaled_xywh
                panel_xywh = [x + x_off, y + y_off, w, h]
                panel_xywh = clamp_xywh(panel_xywh, panel_w, panel_h)

                merged_dets.append({
                    "label": det.label,  # category name
                    "bbox_xywh_panel": [round(v, 3) for v in panel_xywh],
                    "quadrant_index": q,  # 0..3
                    "tile_transform": tile_transforms[q],
                    "source": {
                        "sample_id": str(s.id),
                        "filepath": s.filepath,
                    },
                    # Optional: keep original bbox too (useful for debugging)
                    "bbox_xywh_source_abs": [round(v, 3) for v in abs_xywh],
                })

        # Write outputs
        panel_name = f"panel_{panel_idx:05d}"
        img_path = out / "images" / f"{panel_name}.jpg"
        ann_path = out / "annotations" / f"{panel_name}.json"

        cv2.imwrite(str(img_path), panel_img)

        ann: Dict[str, Any] = {
            "panel_id": panel_name,
            "panel_image": str(img_path),
            "panel_size": {"width": panel_w, "height": panel_h},
            "layout": {"type": "2x2", "tile_size": tile_size},
            "sources": source_meta,
            "tile_transforms": tile_transforms,  # order matches sources/quadrants
            "detections": merged_dets,
        }

        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2)

        if (panel_idx + 1) % 50 == 0:
            print(f"[{panel_idx+1}/{num_panels}] wrote {img_path.name} + {ann_path.name}")

    print("Done.")


if __name__ == "__main__":
    # Example:
    # python generate_coco_comics.py --output_dir /path/to/out --num_panels 1000
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_panels", type=int, default=1000)
    p.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    p.add_argument("--max_source_images", type=int, default=4000)
    p.add_argument("--tile_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--transform_policy", type=str, default="random",
                   choices=["none", "random", "lowres_vs_normal"])
    args = p.parse_args()

    main(
        output_dir=args.output_dir,
        num_panels=args.num_panels,
        split=args.split,
        max_source_images=args.max_source_images,
        tile_size=args.tile_size,
        seed=args.seed,
        transform_policy=args.transform_policy,
    )
