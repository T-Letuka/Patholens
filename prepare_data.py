
import shutil
import random
import hashlib
import logging
import json
from pathlib import Path

from PIL import Image
import pandas as pd




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



CONFIG = {
    "raw_root":         Path("data/raw/Lung_colon_data/lung_colon_image_set"),
    "train_val_folder": "Train and Validation Set",
    "test_folder":      "Test Set",

    "processed_dir":    Path("data/processed"),
    "manifest_path":    Path("data/manifest.csv"),
    "config_path":      Path("data/split_config.json"),

    "val_ratio":        0.15,
    "random_seed":      42,

    "valid_extensions": {".jpg", ".jpeg", ".png", ".tiff"},


    "classes": {
        "colon_aca": {"label": "Colon Adenocarcinoma",         "label_index": 0, "malignant": True,  "site": "colon"},
        "colon_n":   {"label": "Colon Benign",                 "label_index": 1, "malignant": False, "site": "colon"},
        "lung_aca":  {"label": "Lung Adenocarcinoma",          "label_index": 2, "malignant": True,  "site": "lung"},
        "lung_n":    {"label": "Lung Benign",                  "label_index": 3, "malignant": False, "site": "lung"},
        "lung_scc":  {"label": "Lung Squamous Cell Carcinoma", "label_index": 4, "malignant": True,  "site": "lung"},
    },
}



def validate_working_directory() -> None:
    cwd = Path.cwd()
    log.info(f"Running from: {cwd}")


    if cwd.name.lower() == "data":
        raise RuntimeError(
            f"\n\n  You are running this script from inside the data/ folder:"
            f"\n  {cwd}"
            f"\n\n  Please go up one level and run from the project root:"
            f"\n  cd .."
            f"\n  python prepare_dataset.py\n"
        )

   
    raw_root = CONFIG["raw_root"]
    if not raw_root.exists():
        raise FileNotFoundError(
            f"\n\n  Raw dataset not found at:"
            f"\n  {raw_root}"
            f"\n\n  Update CONFIG['raw_root'] to point to your lung_colon_image_set folder.\n"
        )



def validate_structure() -> None:
    log.info("Validating raw dataset structure...")

    raw_root      = CONFIG["raw_root"]
    train_val_dir = raw_root / CONFIG["train_val_folder"]
    test_dir      = raw_root / CONFIG["test_folder"]

    for name, path in [
        (CONFIG["train_val_folder"], train_val_dir),
        (CONFIG["test_folder"],      test_dir),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"\n  Folder not found: {path}"
                f"\n  Must be named exactly '{name}' — capitals and spaces matter.\n"
            )

   
    for top_dir in [train_val_dir, test_dir]:
        for class_folder in CONFIG["classes"]:
            class_path = top_dir / class_folder
            if not class_path.exists():
                raise FileNotFoundError(
                    f"\n  Class folder not found: {class_path}"
                    f"\n  Check spelling exactly.\n"
                )

    log.info("  All folders confirmed.")



def collect_images(folder: Path) -> list:
    exts = CONFIG["valid_extensions"]
    return sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in exts and not p.name.startswith(".")
    ])



def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        log.warning(f"  Corrupt — skipping: {path.name}  ({e})")
        return False



def split_train_val(images: list) -> tuple:
   
    images = list(images)                           
    random.seed(CONFIG["random_seed"])
    random.shuffle(images)

    n_val        = int(len(images) * CONFIG["val_ratio"])
    val_images   = images[:n_val]
    train_images = images[n_val:]
    return train_images, val_images


def compute_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()



def copy_batch(
    images:       list,
    dest_dir:     Path,
    split:        str,
    class_folder: str,
    class_info:   dict,
) -> list:
 
    dest_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for src_path in images:
        dest_path = dest_dir / src_path.name
        shutil.copy2(src_path, dest_path)

        records.append({
            "split":        split,
            "class_folder": class_folder,
            "label":        class_info["label"],
            "label_index":  class_info["label_index"],
            "malignant":    class_info["malignant"],
            "site":         class_info["site"],
            "filename":     src_path.name,
            "image_path":   str(dest_path),
            "md5":          compute_md5(dest_path),
        })

    log.info(f"  {class_folder:<12}  [{split:>5}]  {len(records)} images copied")
    return records


def process_test_set() -> list:
    log.info("─" * 50)
    log.info("TEST SET — copying directly (no splitting)")
    log.info("─" * 50)

    test_src = CONFIG["raw_root"] / CONFIG["test_folder"]
    all_records = []

    for class_folder, class_info in CONFIG["classes"].items():
        src_dir = test_src / class_folder

        images       = collect_images(src_dir)
        valid_images = [p for p in images if is_valid_image(p)]

        n_skipped = len(images) - len(valid_images)
        if n_skipped:
            log.warning(f"  {class_folder}: {n_skipped} corrupt images skipped")

        dest_dir = CONFIG["processed_dir"] / "test" / class_folder
        records  = copy_batch(valid_images, dest_dir, "test", class_folder, class_info)
        all_records.extend(records)

    return all_records


def process_train_val_set() -> list:
    log.info("─" * 50)
    log.info("TRAIN + VAL SET — splitting 85% train / 15% val")
    log.info("─" * 50)

    src_root    = CONFIG["raw_root"] / CONFIG["train_val_folder"]
    all_records = []

    for class_folder, class_info in CONFIG["classes"].items():
        src_dir = src_root / class_folder

        images       = collect_images(src_dir)
        valid_images = [p for p in images if is_valid_image(p)]

        n_skipped = len(images) - len(valid_images)
        if n_skipped:
            log.warning(f"  {class_folder}: {n_skipped} corrupt images skipped")

        train_images, val_images = split_train_val(valid_images)

     
        train_dest = CONFIG["processed_dir"] / "train" / class_folder
        records    = copy_batch(train_images, train_dest, "train", class_folder, class_info)
        all_records.extend(records)

     
        val_dest = CONFIG["processed_dir"] / "val" / class_folder
        records  = copy_batch(val_images, val_dest, "val", class_folder, class_info)
        all_records.extend(records)

    return all_records



def save_manifest(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = df[[
        "split", "class_folder", "label", "label_index",
        "malignant", "site", "filename", "image_path", "md5"
    ]]

    path = CONFIG["manifest_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Manifest saved  →  {path}  ({len(df):,} rows)")
    return df



def save_config() -> None:
    record = {
        "random_seed":      CONFIG["random_seed"],
        "val_ratio":        CONFIG["val_ratio"],
        "train_val_folder": CONFIG["train_val_folder"],
        "test_folder":      CONFIG["test_folder"],
        "classes": {k: v["label"] for k, v in CONFIG["classes"].items()},
        "note": "Delete data/processed/ and re-run to regenerate with new settings."
    }
    path = CONFIG["config_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    log.info(f"Config saved    →  {path}")



def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 68)
    print("  PATHOLENS — DATASET SUMMARY")
    print("=" * 68)

    summary   = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    col_order = [c for c in ["train", "val", "test"] if c in summary.columns]
    summary   = summary[col_order]
    summary["TOTAL"] = summary.sum(axis=1)

    print(summary.to_string())
    print("-" * 68)
    print(f"  Grand total  :  {len(df):,} images")
    print(f"  Per split    :  {df['split'].value_counts().to_dict()}")
    print(f"  Classes      :  {df['label'].nunique()}")
    print("=" * 68)

    expected = {"test": 500, "train": 3825, "val": 675}
    for split_name, exp_count in expected.items():
        split_df = df[df["split"] == split_name]
        for label, count in split_df.groupby("label").size().items():
            if count != exp_count:
                log.warning(
                    f"  Unexpected count — {split_name}/{label}: "
                    f"got {count}, expected {exp_count}"
                )



def main():
    log.info("=" * 55)
    log.info("  PathoLens — Dataset Preparation")
    log.info("=" * 55)

   
    validate_working_directory()

   
    processed_dir = CONFIG["processed_dir"]
    if processed_dir.exists():
        log.warning(
            f"\n  data/processed/ already exists."
            f"\n  This will DELETE and RECREATE it."
            f"\n  Ctrl+C within 5 seconds to cancel.\n"
        )
        import time; time.sleep(5)
        shutil.rmtree(processed_dir)
        log.info("  Old data/processed/ removed.")

    validate_structure()

    test_records = process_test_set()


    train_val_records = process_train_val_set()

  
    df = save_manifest(test_records + train_val_records)
    save_config()

    print_summary(df)

    log.info("Done. Next step: eda")


if __name__ == "__main__":
    main()