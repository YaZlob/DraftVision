from collections.abc import Mapping


def validate_coco_ann(annotation: dict[str, list]) -> None:
    if not isinstance(annotation, Mapping):
        raise ValueError("❌ Error: Top-level structure must be a JSON Object (dict).")

    miss = {"images", "annotations", "categories"}.difference(annotation)
    if miss:
        raise ValueError(
            f"❌ Error: Annotation miss mandatory top-level keys '{miss}'."
        )

    image_ids = set()
    category_ids = set()

    fields = [("id", int), ("name", str)]
    for idx, cat in enumerate(annotation["categories"]):
        if not isinstance(cat, Mapping):
            raise ValueError(
                f"❌ Error: Category entry at index {idx} is not an object."
            )

        for field, expected_type in fields:
            if field not in cat:
                raise ValueError(
                    f"❌ Error: Category entry at index {idx} missing field '{field}'."
                )

            elif not isinstance(cat[field], expected_type):
                raise ValueError(
                    f"❌ Error: Category field '{field}' must be {expected_type.__name__}."
                )

        if "id" in cat:
            category_ids.add(cat["id"])

    fields = [
        ("id", int),
        ("file_name", str),
        ("width", (int, float)),
        ("height", (int, float)),
    ]
    for img in annotation["images"]:
        if not isinstance(img, Mapping):
            raise ValueError(f"❌ Error: expected map, got {img}")

        for field, expected_type in fields:
            if field not in img:
                raise ValueError(
                    f"❌ Error: Image entry {img} missing field '{field}'."
                )

            if not isinstance(img[field], expected_type):
                raise ValueError(
                    f"❌ Error: Image field '{field}' must be {expected_type}. {img}"
                )

        image_ids.add(img["id"])

    fields = [("id", int), ("image_id", int), ("category_id", int), ("bbox", list)]
    for ann in annotation["annotations"]:
        if not isinstance(ann, Mapping):
            raise ValueError(f"❌ Error: Annotation {ann} is not an map.")

        for field, expected_type in fields:
            if field not in ann:
                raise ValueError(f"❌ Error: Annotation {ann} missing field '{field}'.")
            if not isinstance(ann[field], expected_type):
                raise ValueError(
                    f"❌ Error: Annotation field '{field}' must be {expected_type}.\n {ann}"
                )

        if ann["image_id"] not in image_ids:
            raise RuntimeError(
                f"❌ Error: Annotation {ann} points to missing image. Id:{ann['image_id']}."
            )
        if ann["category_id"] not in category_ids:
            raise RuntimeError(
                f"❌ Error: Annotation {ann} points to missing category_id {ann['category_id']}."
            )

        bbox = ann["bbox"]
        if len(bbox) != 4:
            raise ValueError(
                f"❌ Error: Annotation 'bbox' length must be exactly 4 elements [x, y, w, h].\n{ann}."
            )
        if not all(isinstance(coord, (int, float)) for coord in bbox):
            raise ValueError(
                f"❌ Error: Annotation 'bbox' must contain only numerical coordinate values.\n {ann}"
            )
