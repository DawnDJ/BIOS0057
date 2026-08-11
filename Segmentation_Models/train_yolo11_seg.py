import os
import cv2
import numpy as np
import shutil
import json
import random
from pathlib import Path
from ultralytics import YOLO
from multiprocessing import freeze_support
from PIL import Image

# Disable PIL decompression bomb limit for large images
Image.MAX_IMAGE_PIXELS = None


def resize_images_in_dir(input_dir, output_dir=None, max_size=2000):
    """
    Resize all images in input_dir to a maximum dimension (keep aspect ratio).
    If output_dir is None, overwrite originals (use with caution).
    """
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(input_dir, fname)
            img = cv2.imread(src)
            if img is None:
                continue
            h, w = img.shape[:2]
            if max(h, w) <= max_size:
                # Already small enough; copy if different directory
                if output_dir != input_dir:
                    shutil.copy2(src, os.path.join(output_dir, fname))
                continue
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, resized)
            print(f"Resized {fname} from ({w},{h}) to ({new_w},{new_h})")
    print(f"Image resizing completed. Images are in {output_dir}")


def main():
    # ================== Paths ==================
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_IMAGES_DIR = os.path.join(SCRIPT_DIR, "label_json2coco")   # 原始图片和JSON
    # 可选：如果希望保留原图，创建缩略图副本，取消下面一行注释，并修改 TRAIN_IMAGES_DIR
    # TRAIN_IMAGES_DIR = os.path.join(SCRIPT_DIR, "label_json2coco_resized")  # 如果你预先缩放

    TRAIN_ANNOTATIONS = os.path.join(SCRIPT_DIR, "annotations_coco.json")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_yolo_finetuned_full")
    YOLO_DATA_DIR = os.path.join(SCRIPT_DIR, "yolo_dataset_full")

    # ================== Optional: Resize images before training ==================
    # 如果你想缩放图片，取消下面两行注释（会生成缩略图到新文件夹，并自动更新 TRAIN_IMAGES_DIR）
    # RESIZED_DIR = os.path.join(SCRIPT_DIR, "label_json2coco_resized")
    # resize_images_in_dir(SCRIPT_DIR + "/label_json2coco", RESIZED_DIR, max_size=2000)
    # TRAIN_IMAGES_DIR = RESIZED_DIR   # 然后让脚本使用缩放后的图片

    # ================== Training Parameters ==================
    IMGSZ = 640                       # Image size for training
    EPOCHS = 200                      # High upper limit, early stopping will decide actual stop
    BATCH_SIZE = 4                    # Reduced from 8 to lower memory pressure
    CONF_THRESHOLD = 0.1              # Inference confidence threshold (not used in training)
    VAL_SPLIT = 0.1                   # 10% of images used for validation

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(YOLO_DATA_DIR, exist_ok=True)

    # ================== 1. Convert COCO to YOLO format with 9:1 train/val split ==================
    print("Converting COCO annotations to YOLO format with 9:1 train/val split...")
    with open(TRAIN_ANNOTATIONS, 'r') as f:
        coco = json.load(f)

    img_id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    img_id_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    img_files = []
    for img_id, img_file in img_id_to_file.items():
        if img_id in img_id_to_anns:
            img_files.append(img_file)

    random.seed(42)
    random.shuffle(img_files)
    split_idx = int((1 - VAL_SPLIT) * len(img_files))
    train_files = img_files[:split_idx]
    val_files = img_files[split_idx:]

    print(f"Total images: {len(img_files)}, Train: {len(train_files)}, Val: {len(val_files)}")

    images_train_dir = os.path.join(YOLO_DATA_DIR, "images", "train")
    labels_train_dir = os.path.join(YOLO_DATA_DIR, "labels", "train")
    images_val_dir = os.path.join(YOLO_DATA_DIR, "images", "val")
    labels_val_dir = os.path.join(YOLO_DATA_DIR, "labels", "val")
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    def process_files(file_list, images_dir, labels_dir):
        for img_file in file_list:
            img_id = None
            for k, v in img_id_to_file.items():
                if v == img_file:
                    img_id = k
                    break
            if img_id is None:
                continue

            img_path = os.path.join(TRAIN_IMAGES_DIR, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Cannot read {img_path}, skipping")
                continue

            h, w = img.shape[:2]
            shutil.copy(img_path, os.path.join(images_dir, img_file))

            txt_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
            with open(txt_path, 'w') as f_out:
                for ann in img_id_to_anns[img_id]:
                    seg = ann['segmentation'][0]
                    points = np.array(seg).reshape(-1, 2).astype(np.float32)
                    points[:, 0] /= w
                    points[:, 1] /= h
                    line = "0 " + " ".join([f"{p:.6f}" for p in points.flatten()])
                    f_out.write(line + "\n")

    process_files(train_files, images_train_dir, labels_train_dir)
    process_files(val_files, images_val_dir, labels_val_dir)
    print("COCO conversion and split completed.")

    # ================== 2. Create data.yaml ==================
    data_yaml = f"""
path: {YOLO_DATA_DIR}
train: images/train
val: images/val
nc: 1
names: ['butterfly']
"""
    yaml_path = os.path.join(YOLO_DATA_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)
    print(f"data.yaml created at {yaml_path}")

    # ================== 3. Train YOLO11-seg ==================
    print("Loading YOLO11-seg model...")
    model = YOLO("yolo11m-seg.pt")
    print("Training started...")
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=0,
        project=OUTPUT_DIR,
        name="yolo_finetuned_full",
        exist_ok=True,
        verbose=True,
        patience=15,
        augment=True,
        workers=0,               # 禁用多进程，避免并行解码大图导致内存不足
    )
    print("Training completed.")

    print("\nTraining finished. You can now use the trained model to segment any image.")
    print(f"Best model saved at: {os.path.join(OUTPUT_DIR, 'yolo_finetuned_full', 'weights', 'best.pt')}")


if __name__ == '__main__':
    freeze_support()
    main()