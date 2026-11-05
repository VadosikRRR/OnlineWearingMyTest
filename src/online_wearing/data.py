"""Bounded FITVTO streaming/export. Does not download the 100k dataset."""
from io import BytesIO
from itertools import islice
import json
import math
from pathlib import Path

from .contracts import MEASUREMENT_FIELDS, normalize_garment_type

DATASET_ID = "Yuanhao-Harry-Wang/fitvto-100k"


def read_manifest(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _image(value):
    from PIL import Image, ImageOps
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    if isinstance(value, dict):
        payload = value.get("bytes")
        value = BytesIO(payload) if payload is not None else value.get("path")
    if value is None:
        raise ValueError("Dataset image has neither bytes nor path")
    with Image.open(value) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def export_rows(rows, output_dir, *, limit, default_garment_type, garment_types=None):
    """Export exactly limit records, requesting no (limit+1)-th row from the iterator."""
    if not 1 <= limit <= 1000:
        raise ValueError("This evaluation subset must contain between 1 and 1000 rows")
    default_garment_type = normalize_garment_type(default_garment_type)
    garment_types = garment_types or {}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    partial = root / "manifest.jsonl.partial"
    count = 0
    with partial.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(islice(rows, limit)):
            missing = set(("cloth", "person", "target", *MEASUREMENT_FIELDS)) - row.keys()
            if missing:
                raise ValueError(f"Unexpected FITVTO schema; missing {sorted(missing)}. Found {sorted(row)}")
            sample_id = f"{index:06d}"
            folder = root / sample_id
            folder.mkdir(exist_ok=True)
            record = {"id": sample_id, "source_index": index,
                      "garment_type": garment_types.get(sample_id, default_garment_type),
                      "garment_type_source": "override" if sample_id in garment_types else "explicit_default"}
            for key in ("cloth", "person", "target"):
                relative = Path(sample_id) / f"{key}.png"
                path = root / relative
                if not path.exists():
                    temporary = path.with_suffix(".tmp.png")
                    _image(row[key]).save(temporary)
                    temporary.replace(path)
                else:
                    from PIL import Image
                    with Image.open(path) as image:
                        image.verify()
                record[key] = relative.as_posix()
            measurements = {}
            for key in MEASUREMENT_FIELDS:
                value = row[key]
                value = float(value) if value is not None else None
                measurements[key] = value if value is not None and math.isfinite(value) else None
            record["measurements_cm"] = measurements
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
            if count % 100 == 0:
                print(f"Exported FITVTO records: {count}/{limit}", flush=True)
    if count != limit:
        raise ValueError(f"Requested {limit} records but stream ended after {count}; partial manifest retained")
    partial.replace(root / "manifest.jsonl")
    return root / "manifest.jsonl"


def prepare_fitvto_subset(output_dir, *, limit=1000, split="eval", revision="main",
                          default_garment_type="long sleeve top", garment_types=None):
    """Stream a deterministic prefix of eval; persist resolved commit and original cm fields.

    FITVTO has no garment category column. default_garment_type is an explicit
    user assumption, not a category prediction. Per-id overrides are supported.
    Reusing an existing completed subset requires no network access.
    """
    if not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer in [1,1000]")
    if split not in ("train", "eval"):
        raise ValueError("FITVTO split must be train or eval")
    default_garment_type = normalize_garment_type(default_garment_type)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source_file = root / "source.json"
    specification = dict(dataset=DATASET_ID, split=split, limit=limit, requested_revision=revision,
                         default_garment_type=default_garment_type, garment_types=garment_types or {})
    if source_file.exists():
        source = json.loads(source_file.read_text(encoding="utf-8"))
        if any(source.get(key) != value for key, value in specification.items()):
            raise ValueError("Existing subset has different configuration; choose a new output directory")
        manifest = root / "manifest.jsonl"
        if manifest.exists():
            records = read_manifest(manifest)
            if len(records) == limit and all((root / row[key]).is_file()
                                             for row in records for key in ("cloth", "person", "target")):
                return manifest
    else:
        from huggingface_hub import HfApi
        sha = HfApi().dataset_info(DATASET_ID, revision=revision).sha
        source = dict(specification, resolved_revision=sha,
                      measurements_used_for_generation=False,
                      selection="first N rows in repository shard order, without shuffle",
                      image_columns=["cloth", "person", "target"])
        source_file.write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
    from datasets import Image as DatasetImage, load_dataset
    dataset = load_dataset(DATASET_ID, split=split, revision=source["resolved_revision"], streaming=True)
    # Decode via PIL ourselves; avoid a datasets/Pillow/torchcodec decoding dependency.
    for column in ("cloth", "person", "target"):
        dataset = dataset.cast_column(column, DatasetImage(decode=False))
    return export_rows(iter(dataset), root, limit=limit, default_garment_type=default_garment_type,
                       garment_types=garment_types)
