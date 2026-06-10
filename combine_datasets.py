import os
import shutil
from pathlib import Path

DATASETS = [
    "pothole_dataset_1",
    "pothole_dataset_2",
    "pothole_dataset_3",
]

OUTPUT = "pothole_combined"
SPLITS = ["train", "valid", "test"]

def fix_label_file(src_path, dst_path):
    """Copy label file but force class ID to 0 (pothole)"""
    with open(src_path, "r") as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # Force class ID to 0 regardless of original
            parts[0] = "0"
            fixed_lines.append(" ".join(parts) + "\n")

    with open(dst_path, "w") as f:
        f.writelines(fixed_lines)

def combine():
    # Create output folders
    for split in SPLITS:
        os.makedirs(f"{OUTPUT}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT}/{split}/labels", exist_ok=True)

    total_images = 0

    for dataset in DATASETS:
        dataset_path = Path(dataset)

        if not dataset_path.exists():
            print(f"WARNING: {dataset} not found — skipping")
            continue

        print(f"Processing {dataset}...")

        for split in SPLITS:
            img_src = dataset_path / split / "images"
            lbl_src = dataset_path / split / "labels"

            if not img_src.exists():
                print(f"  No {split}/images found — skipping")
                continue

            # Copy images with unique names
            for img_file in img_src.glob("*.*"):
                new_name = f"{dataset}_{img_file.name}"
                dst = f"{OUTPUT}/{split}/images/{new_name}"
                shutil.copy(img_file, dst)
                total_images += 1

            # Fix and copy labels
            if lbl_src.exists():
                for lbl_file in lbl_src.glob("*.txt"):
                    new_name = f"{dataset}_{lbl_file.name}"
                    dst = f"{OUTPUT}/{split}/labels/{new_name}"
                    fix_label_file(lbl_file, dst)

        print(f"  Done!")

    # Create unified data.yaml
    yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: 1
names: ['pothole']
"""
    with open(f"{OUTPUT}/data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\nCombination complete!")
    print(f"Total images combined: {total_images}")
    print(f"Output folder: {OUTPUT}/")
    print(f"All class names standardized to: pothole")
    print(f"data.yaml created successfully!")

combine()