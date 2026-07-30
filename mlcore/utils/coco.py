import json
import numpy as np
import pandas as pd
from PIL import Image
from typing import TypedDict
from pathlib import Path
from collections import defaultdict
from mlcore.core import IMGEXT
from mlcore.data.detection import YoloStyleDataset, validate_coco_ann


class COCO_ANN(TypedDict):
    images: list[dict]
    annotations: list[dict]
    categories: list[dict]


def load_json(path2json: str | Path):
    with open(path2json, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def calculate_coco_statistics(
    annotation_file: str,
    percentiles: list[float] = (10, 25, 50, 75, 95),
    include_relative: bool = False,
) -> dict:

    coco = load_json(annotation_file)
    validate_coco_ann(coco)

    categories = {
        cat["id"]: cat.get("name", str(cat["id"])) for cat in coco.get("categories", [])
    }
    image_sizes = {
        img["id"]: (img.get("width"), img.get("height"))
        for img in coco.get("images", [])
    }

    counts = defaultdict(int)
    images_by_cat = defaultdict(set)

    values = defaultdict(lambda: {"w": [], "h": [], "rel_w": [], "rel_h": []})

    skipped = 0

    for ann in coco.get("annotations", []):
        cat_id = ann.get("category_id")
        img_id = ann.get("image_id")
        bbox = ann.get("bbox")

        w, h = [v for v in bbox[2:]]

        counts[cat_id] += 1
        if img_id is not None:
            images_by_cat[cat_id].add(img_id)

        v = values[cat_id]
        v["w"].append(w)
        v["h"].append(h)

        if include_relative:
            iw, ih = image_sizes.get(img_id, (None, None))

            if iw is not None and ih is not None and iw > 0 and ih > 0:
                rel_w = w / iw
                rel_h = h / ih
                v["rel_w"].append(rel_w)
                v["rel_h"].append(rel_h)

    all_cat_ids = sorted(
        set(categories.keys()) | set(counts.keys()),
        key=lambda x: str(x),
    )

    total_boxes = sum(counts.values())
    total_images = len(image_sizes)
    class_rows, dist_rows = [], []

    for cat_id in all_cat_ids:
        name = categories.get(cat_id, str(cat_id))
        n_boxes = counts.get(cat_id, 0)
        n_images = len(images_by_cat.get(cat_id, set()))

        percent_of_boxes = 100.0 * n_boxes / total_boxes if total_boxes > 0 else 0.0
        percent_of_images = 100.0 * n_images / total_images if total_images > 0 else 0.0

        row = {
            "category_id": cat_id,
            "category_name": name,
            "num_boxes": n_boxes,
            "num_images_with_class": n_images,
            "percent_of_boxes": percent_of_boxes,
            "percent_of_images": percent_of_images,
        }

        v = values.get(cat_id)

        if v and v["w"]:
            w = np.asarray(v["w"], dtype=np.float64)
            h = np.asarray(v["h"], dtype=np.float64)
            area = w * h

            row.update(
                {
                    "width_min": np.min(w),
                    "width_max": np.max(w),
                    "width_mean": np.mean(w),
                    "width_std": np.std(w),
                    "height_min": np.min(h),
                    "height_max": np.max(h),
                    "height_mean": np.mean(h),
                    "height_std": np.std(h),
                    "area_min": np.min(area),
                    "area_max": np.max(area),
                    "area_mean": np.mean(area),
                    "area_std": np.std(area),
                }
            )

            valid_aspect_mask = h > 0
            aspect = w[valid_aspect_mask] / h[valid_aspect_mask]
            aspect = aspect[np.isfinite(aspect)]

            if aspect.size > 0:
                row["aspect_ratio_mean"] = np.mean(aspect)
                row["aspect_ratio_std"] = np.std(aspect)
            else:
                row["aspect_ratio_mean"] = np.nan
                row["aspect_ratio_std"] = np.nan

            for p in percentiles:
                p_tag = str(p).replace(".", "_")

                row[f"width_p{p_tag}"] = np.percentile(w, p)
                row[f"height_p{p_tag}"] = np.percentile(h, p)
                row[f"area_p{p_tag}"] = np.percentile(area, p)

                if aspect.size > 0:
                    row[f"aspect_ratio_p{p_tag}"] = np.percentile(aspect, p)
                else:
                    row[f"aspect_ratio_p{p_tag}"] = np.nan

            if include_relative and v["rel_w"] and v["rel_h"]:
                rel_w = np.asarray(v["rel_w"], dtype=np.float64)
                rel_h = np.asarray(v["rel_h"], dtype=np.float64)

                row.update(
                    {
                        "num_boxes_with_relative_size": len(rel_w),
                        "relative_width_mean": np.mean(rel_w),
                        "relative_height_mean": np.mean(rel_h),
                        "relative_width_std": np.std(rel_w),
                        "relative_height_std": np.std(rel_h),
                        "relative_width_min": np.min(rel_w),
                        "relative_width_max": np.max(rel_w),
                        "relative_height_min": np.min(rel_h),
                        "relative_height_max": np.max(rel_h),
                    }
                )

                for p in percentiles:
                    p_tag = str(p).replace(".", "_")

                    row[f"relative_width_p{p_tag}"] = np.percentile(rel_w, p)
                    row[f"relative_height_p{p_tag}"] = np.percentile(rel_h, p)

        class_rows.append(row)

        dist_rows.append(
            {
                "category_id": cat_id,
                "category_name": name,
                "num_boxes": n_boxes,
                "num_images_with_class": n_images,
                "percent_of_boxes": percent_of_boxes,
                "percent_of_images": percent_of_images,
            }
        )

    import pandas as pd

    class_stats = (
        pd.DataFrame(class_rows)
        .sort_values("num_boxes", ascending=False)
        .reset_index(drop=True)
    )

    distribution = (
        pd.DataFrame(dist_rows)
        .sort_values("num_boxes", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "class_stats": class_stats,
        "distribution": distribution,
        "total_boxes": total_boxes,
        "total_categories": len(all_cat_ids),
        "total_images": total_images,
        "skipped_annotations": skipped,
    }


def _percentile_column(prefix: str, p: float) -> str:
    """
    Convert percentile value to column name used in the stats DataFrame.

    Examples:
        width, 50   -> width_p50
        height, 95  -> height_p95
        width, 97.5 -> width_p97_5
    """
    return f"{prefix}_p{str(p).replace('.', '_')}"


def pretty_print_coco_statistics(
    stats: dict,
    top_k: int = 20,
    percentiles=(25, 50, 75, 95),
) -> None:
    """
    Pretty-print COCO statistics.

    Parameters
    ----------
    stats : dict
        Output from calculate_coco_statistics().
    top_k : int
        Print only top K classes by number of boxes.
        If None, prints all classes.
    percentiles : tuple/list
        Percentile columns to show in the per-class table.
    """

    total_images = stats.get("total_images", 0) or 0
    total_boxes = stats.get("total_boxes", 0) or 0
    total_categories = stats.get("total_categories", 0) or 0
    skipped_annotations = stats.get("skipped_annotations", 0) or 0

    print("=" * 100)
    print("COCO DATASET SUMMARY")
    print("=" * 100)
    print(f"Total images:           {total_images:,}")
    print(f"Total boxes:            {total_boxes:,}")
    print(f"Total categories:       {total_categories:,}")
    print(f"Skipped annotations:    {skipped_annotations:,}")

    print()
    print("=" * 100)
    print("CLASS DISTRIBUTION")
    print("=" * 100)

    dist = stats.get("distribution", pd.DataFrame()).copy()

    if top_k:
        dist = dist.head(top_k)

    if dist.empty:
        print("No class distribution data found.")
    else:
        dist_cols = [
            c
            for c in [
                "category_id",
                "category_name",
                "num_boxes",
                "percent_of_boxes",
                "num_images_with_class",
                "percent_of_images",
            ]
            if c in dist.columns
        ]

        dist_formatters = {}

        for col in dist_cols:
            if col in ("num_boxes", "num_images_with_class"):
                dist_formatters[col] = lambda x: f"{int(x):,}" if pd.notna(x) else ""
            elif col in ("percent_of_boxes", "percent_of_images"):
                dist_formatters[col] = lambda x: f"{x:.2f}%" if pd.notna(x) else ""

        print(dist[dist_cols].to_string(index=False, formatters=dist_formatters))

    print()
    print("=" * 100)
    print("PER-CLASS BBOX SIZE STATISTICS")
    print("=" * 100)

    class_stats = stats.get("class_stats", pd.DataFrame()).copy()

    if top_k:
        class_stats = class_stats.head(top_k)

    if class_stats.empty:
        print("No per-class statistics found.")
        return

    percentile_cols = []

    for p in percentiles:
        w_col = _percentile_column("width", p)
        h_col = _percentile_column("height", p)

        if w_col in class_stats.columns:
            percentile_cols.append(w_col)

        if h_col in class_stats.columns:
            percentile_cols.append(h_col)

    base_cols = [
        c
        for c in ["category_name", "num_boxes", "percent_of_boxes"]
        if c in class_stats.columns
    ]

    if percentile_cols:
        selected_cols = base_cols + percentile_cols
    else:
        fallback_cols = [
            c
            for c in [
                "width_mean",
                "height_mean",
                "width_std",
                "height_std",
            ]
            if c in class_stats.columns
        ]
        selected_cols = base_cols + fallback_cols

    class_formatters = {}

    for col in selected_cols:
        if col == "num_boxes":
            class_formatters[col] = lambda x: f"{int(x):,}" if pd.notna(x) else ""
        elif col == "percent_of_boxes":
            class_formatters[col] = lambda x: f"{x:.2f}%" if pd.notna(x) else ""
        elif col.startswith(
            (
                "width",
                "height",
                "area",
                "aspect_ratio",
                "relative",
            )
        ):
            class_formatters[col] = lambda x: f"{x:,.2f}" if pd.notna(x) else ""

    print(
        class_stats[selected_cols].to_string(index=False, formatters=class_formatters)
    )


def yolo_to_coco(
    img_dir: str | Path,
    ann_dir: str | Path,
    output_dir: str | Path | None = None,
    classes: dict[int, str] | None = None,
    target: str = "val",
):

    img_dir = Path(img_dir)
    ann_dir = Path(ann_dir)

    if not (img_dir.is_dir() or ann_dir.is_dir()):
        raise ValueError()

    image_id = annotation_id = 1
    categories = set()
    images, annotations = [], []

    image_wo_ann = []
    for image_path in img_dir.iterdir():
        if not image_path.suffix in IMGEXT:
            continue

        # Reads only HEAD, not full image
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as err:
            msg = (
                f"Couldn't read {image_path.name} head, mb file was damaged, err: {err}"
            )
            print(msg)
            continue

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        ann_name = image_path.stem + ".txt"
        ann_path = ann_dir / ann_name
        if ann_path.is_file():
            labels, boxes = YoloStyleDataset._load_ann(ann_path)
            categories.update(labels)
            for label, box in zip(labels, boxes):
                cx, cy, w, h = box
                bw, bh = w * width, h * height
                xmin = cx * width - bw / 2
                ymin = cy * height - bh / 2

                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [
                            round(xmin, 2),
                            round(ymin, 2),
                            round(bw, 2),
                            round(bh, 2),
                        ],
                        "area": round(bw * bh, 2),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        else:
            image_wo_ann.append(image_path.name)

        image_id += 1

    min_id = min(categories)
    max_id = max(categories)
    expected_sequence = set(range(min_id, max_id + 1))
    missing_ids = expected_sequence - set(categories)

    if min_id != 0:
        raise ValueError(f"category id must start from 0, in your case from {min_id}")
    if missing_ids:
        raise ValueError(f"Categories missed ids: {missing_ids}")

    if classes:
        miss_categories = set(classes).difference(categories)
        if miss_categories:
            raise ValueError(f"Dataset miss categories: {miss_categories}")
        categories = [
            {"id": k, "name": v, "supercategory": None} for k, v in classes.items()
        ]
    else:
        categories = [
            {"id": i, "name": f"cls_{i}", "supercategory": None} for i in categories
        ]

    assert (
        isinstance(categories, list)
        and len(categories) >= 1
        and all([isinstance(data, dict) for data in categories])
    )

    coco_ann = COCO_ANN(images=images, annotations=annotations, categories=categories)
    validate_coco_ann(coco_ann)

    output_dir = ann_dir.parent if output_dir is None else Path(output_dir)
    with open(output_dir / f"{target}.json", "w") as f:
        json.dump(coco_ann, f, indent=4)

    print(f"Successfully converted dataset! COCO file saved to: {output_dir}")
