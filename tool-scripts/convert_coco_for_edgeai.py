#!/usr/bin/env python3
"""
convert_coco_for_edgeai.py

Convert a COCO dataset (images + COCO JSON) into the structure required by
TI Edge AI Studio Model Composer:

    out_dir/
      images/
      annotations/
        instances.json
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_images_and_build_map(
    images_info: List[Dict],
    source_image_dirs: List[Path],
    dest_images_dir: Path
) -> Tuple[Dict[int, str], List[Tuple[int, str]]]:
    id_to_basename: Dict[int, str] = {}
    missing: List[Tuple[int, str]] = []
    for img in images_info:
        img_id = img.get("id")
        file_name = img.get("file_name")
        if img_id is None or not file_name:
            missing.append((img_id, file_name))
            continue
        found_path = None
        candidate_paths = [Path(file_name)]
        candidate_paths += [d / Path(file_name).name for d in source_image_dirs]
        for cand in candidate_paths:
            if cand.is_file():
                found_path = cand
                break
        if not found_path:
            missing.append((img_id, file_name))
            continue
        dest_path = dest_images_dir / found_path.name
        if dest_path.exists():
            try:
                if dest_path.samefile(found_path):
                    new_name = dest_path.name
                else:
                    new_name = f"{img_id}_{found_path.name}"
                    dest_path = dest_images_dir / new_name
                    shutil.copy2(found_path, dest_path)
            except Exception:
                new_name = f"{img_id}_{found_path.name}"
                dest_path = dest_images_dir / new_name
                shutil.copy2(found_path, dest_path)
        else:
            shutil.copy2(found_path, dest_path)
            new_name = dest_path.name
        id_to_basename[img_id] = new_name
    return id_to_basename, missing

def build_output_coco(
    original_coco: Dict,
    id_to_basename: Dict[int, str],
    keep_missing_images: bool = False
) -> Dict:
    out: Dict = {}
    for k in ("info", "licenses", "categories"):
        if k in original_coco:
            out[k] = original_coco[k]

    new_images: List[Dict] = []
    valid_image_ids = set(id_to_basename.keys())
    for img in original_coco.get("images", []):
        img_id = img.get("id")
        if img_id in id_to_basename:
            new_img = dict(img)
            new_img["file_name"] = id_to_basename[img_id]
            new_images.append(new_img)
        else:
            if keep_missing_images:
                new_images.append(img)
    out["images"] = new_images

    new_annotations: List[Dict] = []
    for ann in original_coco.get("annotations", []):
        if ann.get("image_id") in valid_image_ids:
            new_annotations.append(ann)
    out["annotations"] = new_annotations

    return out

def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO dataset into Edge AI Studio Model Composer layout."
    )
    parser.add_argument("--coco_json", required=True, help="Path to source COCO JSON (instances*.json)")
    parser.add_argument("--images_dir", required=True, nargs="+", help="One or more paths to folders containing the images referenced by the COCO JSON.")
    parser.add_argument("--out_dir", required=True, help="Destination output directory (will be created).")
    parser.add_argument("--keep-missing-images", action="store_true", help="If set, images missing from the provided image dirs will still be listed in instances.json.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    args = parser.parse_args()

    coco_path = Path(args.coco_json)
    image_dirs = [Path(d) for d in args.images_dir]
    out_dir = Path(args.out_dir)
    dest_images = out_dir / "images"
    dest_ann_dir = out_dir / "annotations"
    dest_ann_file = dest_ann_dir / "instances.json"

    if not coco_path.is_file():
        raise SystemExit(f"COCO JSON not found: {coco_path}")

    for d in image_dirs:
        if not d.exists() and not args.quiet:
            print(f"Warning: image dir does not exist: {d}")

    ensure_dir(dest_images)
    ensure_dir(dest_ann_dir)

    if not args.quiet:
        print(f"Loading COCO JSON from {coco_path}...")
    coco = load_json(coco_path)

    images_info = coco.get("images", [])
    if not images_info:
        raise SystemExit("No 'images' array found in COCO JSON.")

    if not args.quiet:
        print(f"Found {len(images_info)} image entries in COCO JSON. Copying images...")

    id_to_basename, missing = copy_images_and_build_map(images_info, image_dirs, dest_images)

    if missing and not args.quiet:
        print(f"Warning: {len(missing)} images referenced by the JSON were not found in provided search paths. They will be excluded from the output unless --keep-missing-images is used.")
        for idx, (img_id, fname) in enumerate(missing[:10]):
            print(f"  missing[{idx}]: id={img_id} file_name={fname}")
        if len(missing) > 10 and not args.quiet:
            print(f"  ... and {len(missing)-10} more")

    out_coco = build_output_coco(coco, id_to_basename, keep_missing_images=args.keep_missing_images)

    with dest_ann_file.open("w", encoding="utf-8") as f:
        json.dump(out_coco, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        n_images = len(out_coco.get("images", []))
        n_annotations = len(out_coco.get("annotations", []))
        print(f"Done. Wrote {n_images} images into {dest_images} and annotations to {dest_ann_file}.")
        print(f"Output instances.json contains {n_images} images and {n_annotations} annotations.")

if __name__ == "__main__":
    main()

