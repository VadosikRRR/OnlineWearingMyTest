"""Resumable paired evaluation: the target image is used only by metrics."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from .data import read_manifest


def image_metrics(prediction, target):
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity
    with Image.open(target) as image:
        truth = np.array(image.convert("RGB").resize(prediction.size, Image.Resampling.LANCZOS))
    predicted = np.array(prediction.convert("RGB"))
    mse = float(np.square((predicted.astype(np.float64) - truth.astype(np.float64)) / 255).mean())
    return {"mae": float(np.abs(predicted.astype(np.float32) - truth).mean() / 255),
            "psnr_db": float(-10 * np.log10(max(mse, 1e-12))),
            "ssim": float(structural_similarity(truth, predicted, channel_axis=-1, data_range=255))}


def evaluate_subset(pipeline, manifest_path, output_dir, *, limit=1000, seed=42,
                    save_debug_first=3, fail_fast=False, metric_fn=None):
    """Resume successful samples, retry failures, never hide failed sample counts."""
    if not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer in [1,1000]")
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    rows = read_manifest(manifest_path)[:limit]
    if not rows:
        raise ValueError("Empty evaluation manifest")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs in manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                     "seed": seed, "weights": pipeline.weights.as_dict(),
                     "sampling_steps": pipeline.sampling_steps, "repaint_repeats": pipeline.repaint_repeats,
                     "sci_precision": pipeline.sci_precision, "device": str(pipeline.device)}
    config_file = output_dir / "run_config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != configuration:
        raise ValueError("Run configuration changed; use a new output directory to avoid mixing results")
    config_file.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    log = output_dir / "metrics.jsonl"
    completed = {}
    if log.exists():
        for row in read_manifest(log):
            completed[row["id"]] = row
    metric_fn = metric_fn or image_metrics
    for index, record in enumerate(rows):
        sample_id = record["id"]
        # Ensure manifest IDs cannot escape the output root.
        if not sample_id.isdigit():
            raise ValueError(f"Invalid sample ID {sample_id!r}")
        folder = output_dir / sample_id
        if (completed.get(sample_id, {}).get("status") == "ok" and (folder / "result.png").is_file()):
            continue
        started = time.perf_counter()
        report = {"id": sample_id, "garment_type": record["garment_type"],
                  "garment_type_source": record.get("garment_type_source"),
                  "measurements_cm": record.get("measurements_cm"), "seed": seed + index}
        try:
            # Crucially, do not provide target/ground-truth pixels to any model.
            result = pipeline(manifest_path.parent / record["cloth"],
                              manifest_path.parent / record["person"], record["garment_type"], seed=seed + index)
            result.save(folder, debug=index < save_debug_first)
            report.update(metric_fn(result.image, manifest_path.parent / record["target"]))
            report["status"] = "ok"
        except Exception as error:
            report.update(status="error", error_type=type(error).__name__, error=str(error))
            if fail_fast:
                report["seconds"] = time.perf_counter() - started
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
                raise
        report["seconds"] = time.perf_counter() - started
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
        completed[sample_id] = report
        print(f"[{index + 1}/{len(rows)}] {sample_id}: {report['status']}", flush=True)
    current = [completed[key] for key in ids]
    successes = [row for row in current if row["status"] == "ok"]
    summary = {"requested": len(rows), "succeeded": len(successes),
               "failed": len(current) - len(successes),
               "metrics_scope": "successful samples only; paired full-image metrics, not physical fit accuracy",
               "errors": dict(Counter(row.get("error_type") for row in current if row["status"] != "ok")),
               "type_sources": dict(Counter(row.get("garment_type_source", "unknown") for row in current))}
    for key in ("mae", "psnr_db", "ssim"):
        summary[key] = sum(row[key] for row in successes) / len(successes) if successes else None
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
