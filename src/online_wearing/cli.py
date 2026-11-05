"""Small CLI; heavyweight dependencies are imported only by the chosen command."""
import argparse
import json
from pathlib import Path

from .contracts import Weights


def main():
    parser = argparse.ArgumentParser(description="OnlineWearing KGI baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Stream and export at most 1000 FITVTO records")
    prepare.add_argument("output")
    prepare.add_argument("--limit", type=int, default=1000)
    prepare.add_argument("--split", choices=["train", "eval"], default="eval")
    prepare.add_argument("--revision", default="main")
    prepare.add_argument("--garment-type", required=True)
    prepare.add_argument("--garment-types", help="JSON mapping of subset IDs (000000 etc.) to category strings")
    for name in ("infer", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--weights", required=True, help="JSON with the six Weights fields")
        command.add_argument("--device", default="cpu")
        command.add_argument("--precision", choices=["float32", "float16"], default="float32")
        command.add_argument("--sampling-steps", type=int, default=50)
        command.add_argument("--repaint-repeats", type=int, default=10)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--output", required=True)
        if name == "infer":
            command.add_argument("--cloth", required=True)
            command.add_argument("--person", required=True)
            command.add_argument("--garment-type", required=True)
        else:
            command.add_argument("--manifest", required=True)
            command.add_argument("--limit", type=int, default=1000)
            command.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare_fitvto_subset
        overrides = json.loads(Path(args.garment_types).read_text()) if args.garment_types else None
        print(prepare_fitvto_subset(args.output, limit=args.limit, split=args.split, revision=args.revision,
                                   default_garment_type=args.garment_type, garment_types=overrides))
        return
    from .pipeline import OnlineWearingPipeline
    weights = Weights(**json.loads(Path(args.weights).read_text()))
    pipeline = OnlineWearingPipeline(weights, device=args.device, sci_precision=args.precision,
                                     sampling_steps=args.sampling_steps, repaint_repeats=args.repaint_repeats)
    if args.command == "infer":
        result = pipeline(args.cloth, args.person, args.garment_type, seed=args.seed)
        print(result.save(args.output))
    else:
        from .evaluation import evaluate_subset
        print(json.dumps(evaluate_subset(pipeline, args.manifest, args.output, limit=args.limit,
                                         seed=args.seed, fail_fast=args.fail_fast), indent=2))


if __name__ == "__main__":
    main()
