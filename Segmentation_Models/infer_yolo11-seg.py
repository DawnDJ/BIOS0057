import os
import cv2
import numpy as np
import json
from ultralytics import YOLO

# ================== Paths ==================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGES_DIR = os.path.join(SCRIPT_DIR, "label_json2coco")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "segmented_individuals_yolo_global_v2")
MODEL_PATH = os.path.join(SCRIPT_DIR, "output_yolo_finetuned_full", "yolo_finetuned_full", "weights", "best.pt")

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMGSZ = 1280
OVERLAP_THRESHOLD = 0.3
PADDING = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded.")

json_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith('.json')]
json_files.sort()
print(f"Found {len(json_files)} JSON files.")

total_instances = 0

def compute_mask_overlap(mask1, mask2):

    inter = np.logical_and(mask1, mask2).sum()
    area2 = mask2.sum()
    return inter / area2 if area2 > 0 else 0.0

for json_file in json_files:
    base_name = os.path.splitext(json_file)[0]
    img_path = os.path.join(INPUT_IMAGES_DIR, base_name + '.jpg')
    if not os.path.exists(img_path):
        print(f"Warning: Image {img_path} not found, skipping JSON {json_file}")
        continue

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Warning: Cannot read {img_path}, skipping")
        continue

    with open(os.path.join(INPUT_IMAGES_DIR, json_file), 'r') as f:
        data = json.load(f)

    shapes = data.get('shapes', [])
    if not shapes:
        print(f"No shapes in {json_file}, skipping")
        continue

    print(f"\nProcessing {json_file} ({len(shapes)} instances)")

    # Run inference on full image
    results = model(img_bgr, imgsz=IMGSZ, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    has_pred = (len(results) > 0 and results[0].masks is not None)

    if has_pred:
        det = results[0]
        pred_boxes = det.boxes.xyxy.cpu().numpy()          # (N,4) in original coords
        pred_masks = det.masks.data.cpu().numpy()          # (N, H_pred, W_pred)
        pred_scores = det.boxes.conf.cpu().numpy()
    else:
        pred_boxes = np.array([])
        pred_masks = np.array([])

    img_out_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(img_out_dir, exist_ok=True)

    vis_img = img_bgr.copy()

    for idx, shape in enumerate(shapes):
        if shape['shape_type'] != 'polygon':
            continue

        points = np.array(shape['points'], dtype=np.float32)
        # Full-size polygon mask (for fallback and final crop)
        poly_mask_full = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)
        cv2.fillPoly(poly_mask_full, [points.astype(np.int32)], 1)
        poly_mask_full_bool = poly_mask_full.astype(bool)

        # Resize polygon mask to IMGSZ for matching
        poly_mask_resized = cv2.resize(poly_mask_full.astype(np.uint8), (IMGSZ, IMGSZ), interpolation=cv2.INTER_NEAREST)
        poly_mask_resized_bool = poly_mask_resized.astype(bool)

        # Bounding box of polygon (original coordinates)
        x1 = int(np.min(points[:, 0]))
        y1 = int(np.min(points[:, 1]))
        x2 = int(np.max(points[:, 0]))
        y2 = int(np.max(points[:, 1]))

        if has_pred and len(pred_boxes) > 0:
            best_overlap = 0.0
            best_idx = -1
            for i, pred_mask in enumerate(pred_masks):
                # Resize prediction mask to IMGSZ×IMGSZ
                pred_mask_resized = cv2.resize(pred_mask.astype(np.float32),
                                               (IMGSZ, IMGSZ),
                                               interpolation=cv2.INTER_LINEAR)
                pred_mask_bool = (pred_mask_resized > 0.5)
                overlap = compute_mask_overlap(pred_mask_bool, poly_mask_resized_bool)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i

            if best_idx != -1 and best_overlap >= OVERLAP_THRESHOLD:
                # Use matched prediction mask (keep in original prediction size)
                pred_mask = pred_masks[best_idx]   # (H_pred, W_pred)
                # Map polygon bbox to prediction mask coordinates
                # We need the letterbox scaling, but we can approximate by scaling.
                # More accurate: get the actual input size from the model (but we can compute scale)
                # Simple scaling: use IMGSZ as reference
                scale_w_pred = pred_mask.shape[1] / IMGSZ
                scale_h_pred = pred_mask.shape[0] / IMGSZ
                bbox_x1 = int(x1 * (img_bgr.shape[1] / IMGSZ) * scale_w_pred)
                bbox_y1 = int(y1 * (img_bgr.shape[0] / IMGSZ) * scale_h_pred)
                bbox_x2 = int(x2 * (img_bgr.shape[1] / IMGSZ) * scale_w_pred)
                bbox_y2 = int(y2 * (img_bgr.shape[0] / IMGSZ) * scale_h_pred)
                # Clamp to mask bounds
                bbox_x1 = max(0, min(pred_mask.shape[1], bbox_x1))
                bbox_y1 = max(0, min(pred_mask.shape[0], bbox_y1))
                bbox_x2 = max(bbox_x1, min(pred_mask.shape[1], bbox_x2))
                bbox_y2 = max(bbox_y1, min(pred_mask.shape[0], bbox_y2))
                # Crop prediction mask region
                pred_mask_region = pred_mask[bbox_y1:bbox_y2, bbox_x1:bbox_x2]  # float
                # Determine final crop size with padding
                crop_w = min(img_bgr.shape[1], x2 + PADDING) - max(0, x1 - PADDING)
                crop_h = min(img_bgr.shape[0], y2 + PADDING) - max(0, y1 - PADDING)
                if crop_w > 0 and crop_h > 0 and pred_mask_region.size > 0:
                    mask_resized = cv2.resize(pred_mask_region,
                                              (crop_w, crop_h),
                                              interpolation=cv2.INTER_LINEAR)
                    mask_bool = (mask_resized > 0.5)
                else:
                    mask_bool = poly_mask_full_bool
            else:
                # Fallback to polygon mask
                mask_bool = poly_mask_full_bool
        else:
            mask_bool = poly_mask_full_bool

        # Crop with padding
        x1_pad = max(0, x1 - PADDING)
        y1_pad = max(0, y1 - PADDING)
        x2_pad = min(img_bgr.shape[1], x2 + PADDING)
        y2_pad = min(img_bgr.shape[0], y2 + PADDING)

        cropped_img = img_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
        mask_cropped = mask_bool[y1_pad:y2_pad, x1_pad:x2_pad]

        # Build transparent PNG
        h_crop, w_crop = cropped_img.shape[:2]
        composite = np.zeros((h_crop, w_crop, 4), dtype=np.uint8)
        if mask_cropped.any():
            composite[mask_cropped, :3] = cropped_img[mask_cropped]
            composite[mask_cropped, 3] = 255
        else:
            # Fallback: use polygon mask directly
            mask_cropped2 = poly_mask_full_bool[y1_pad:y2_pad, x1_pad:x2_pad]
            composite[mask_cropped2, :3] = cropped_img[mask_cropped2]
            composite[mask_cropped2, 3] = 255

        out_path = os.path.join(img_out_dir, f"instance_{idx+1:03d}.png")
        cv2.imwrite(out_path, composite)
        total_instances += 1

        # Draw polygon on visualization
        pts = points.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imwrite(os.path.join(img_out_dir, "visualization.jpg"), vis_img)
    print(f"  Saved {len(shapes)} instances to {img_out_dir}")

print(f"\nProcessing complete. Total instances extracted: {total_instances}")
print(f"Output directory: {OUTPUT_DIR}")