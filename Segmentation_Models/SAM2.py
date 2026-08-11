import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os

# ------------------ 1. Load YOLO format annotation file ------------------
def load_yolo_boxes(txt_path, img_shape):
    """
    Read YOLO format txt file, return list of bounding boxes in pixel coordinates.
    YOLO format: class_id center_x center_y width height (normalized, 0~1)
    Output: [[x1, y1, x2, y2], ...] pixel coordinates
    """
    h, w = img_shape[:2]
    boxes = []
    if not Path(txt_path).exists():
        return boxes
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            # Clip boundaries to prevent out-of-range
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32)


# ------------------ 2. Save single object (transparent background) ------------------
def save_object_cropped(image_bgr, mask, box, output_path, padding=10):
    """
    Crop the object region (with padding) based on mask and original bounding box,
    generate transparent background PNG.
    image_bgr: original image (BGR)
    mask: boolean mask of original size (H, W)
    box: original bounding box [x1, y1, x2, y2] (to determine crop region)
    padding: extra pixels around the bounding box
    """
    x1, y1, x2, y2 = map(int, box)
    # Expand crop region to avoid cutting the object
    crop_x1 = max(0, x1 - padding)
    crop_y1 = max(0, y1 - padding)
    crop_x2 = min(image_bgr.shape[1], x2 + padding)
    crop_y2 = min(image_bgr.shape[0], y2 + padding)

    # Crop image and mask
    cropped_img = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]

    # Create transparent background image
    h_crop, w_crop = cropped_mask.shape
    composite = np.zeros((h_crop, w_crop, 4), dtype=np.uint8)
    composite[cropped_mask, :3] = cropped_img[cropped_mask]
    composite[cropped_mask, 3] = 255
    cv2.imwrite(str(output_path), composite)
    del cropped_img, cropped_mask, composite
    gc.collect()


# ------------------ 3. Process a single image ------------------
def extract_objects_from_image(image_path, txt_path, output_dir, predictor, max_size=2000):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Failed to read image: {image_path}")
        return
    orig_h, orig_w = image_bgr.shape[:2]

    boxes = load_yolo_boxes(txt_path, (orig_h, orig_w))
    if len(boxes) == 0:
        print(f"No bounding box found: {txt_path}")
        return

    # Rescaling
    scale = min(max_size / max(orig_h, orig_w), 1.0)
    use_scale = scale < 1.0
    if use_scale:
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        image_small = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
        boxes_scaled = boxes * scale
        predictor.set_image(image_rgb)
        # Segment each box
        for i, box_scaled in enumerate(boxes_scaled):
            input_box = np.array(box_scaled).reshape(1, 4)
            mask, _, _ = predictor.predict(box=input_box, multimask_output=False)
            mask_small = mask[0, :, :].astype(bool)
            # Upscale mask back to original size
            mask_orig = cv2.resize(mask_small.astype(np.uint8), (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            out_path = output_dir / f"{image_path.stem}_box{i + 1:03d}.png"
            save_object_cropped(image_bgr, mask_orig, boxes[i], out_path, padding=20)
            torch.cuda.empty_cache()
            gc.collect()
        print(f"Extracted {len(boxes)} object(s) from: {image_path.name}")
    else:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        for i, box in enumerate(boxes):
            input_box = np.array(box).reshape(1, 4)
            mask, _, _ = predictor.predict(box=input_box, multimask_output=False)
            mask_orig = mask[0, :, :].astype(bool)
            out_path = output_dir / f"{image_path.stem}_box{i + 1:03d}.png"
            save_object_cropped(image_bgr, mask_orig, box, out_path, padding=20)
            torch.cuda.empty_cache()
            gc.collect()
        print(f"Extracted {len(boxes)} object(s) from: {image_path.name}")


# ------------------ 4. Batch processing main function ------------------
def batch_extract_objects(input_dir, output_dir, config_path, checkpoint_path, max_size=2000):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2 model (device: {device})...")
    sam2_model = build_sam2(config_path, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(input_dir.glob(ext))
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for img_path in tqdm(image_paths, desc="Processing images"):
        txt_path = img_path.with_suffix('.txt')
        extract_objects_from_image(img_path, txt_path, output_dir, predictor, max_size=max_size)

# ------------------------------------
if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))

    INPUT_DIR = os.path.join(script_dir, "input_images_SAM2_2")
    OUTPUT_DIR = os.path.join(script_dir, "extracted_objects_SAM2_2")
    CONFIG_PATH = os.path.join(script_dir, "models", "sam2.1_hiera_l.yaml")
    CHECKPOINT_PATH = os.path.join(script_dir, "models", "sam2.1_hiera_large.pt")
    MAX_SIZE = 2000

    batch_extract_objects(INPUT_DIR, OUTPUT_DIR, CONFIG_PATH, CHECKPOINT_PATH, max_size=MAX_SIZE)