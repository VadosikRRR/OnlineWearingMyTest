"""End-to-end raw-image KGI inference. No dataset annotations are required."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from .checkpoints import load_weights
from .conditioning import draw_cloth, draw_skeleton
from .contracts import (Weights, WIDTH, HEIGHT, CONTOUR_EDGES, SYMMETRY_EDGES,
                        POSE_EDGES, normalize_garment_type)
from .preprocessing import (GarmentLandmarks, HumanPose, HumanParser, PreparedInputs,
                            prepare_image, agnostic_conditions)
from .warping import compose_conditions


class Config(dict):
    """Minimal inference configuration interface required by KGI's RePaint code."""
    def __getattr__(self, name):
        return self.get(name)

    def pget(self, path, default=None):
        value = self
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value


def adjacency(count, edges):
    matrix = torch.eye(count)
    for left, right in edges:
        matrix[left, right] = matrix[right, left] = 1
    return matrix / matrix.sum(dim=1, keepdim=True)


def one_hot(labels, device):
    tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64).copy()).to(device)
    return F.one_hot(tensor, 13).permute(2, 0, 1).unsqueeze(0).float()


def image_tensor(rgb, device):
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(device).float() / 127.5 - 1


def point_overlay(rgb, points):
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    for index, (x, y) in enumerate(points * [WIDTH, HEIGHT]):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="red")
        draw.text((x + 5, y), str(index), fill="blue", stroke_width=1, stroke_fill="white")
    return im


@dataclass
class TryOnResult:
    image: Image.Image
    artifacts: dict[str, Image.Image]
    metadata: dict

    def save(self, directory, *, debug=True):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.image.save(directory / "result.png")
        if debug:
            for name, artifact in self.artifacts.items():
                artifact.save(directory / f"{name}.png")
        (directory / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return directory / "result.png"


class OnlineWearingPipeline:
    """KGI: HRNet + OpenPose + SCHP -> KG -> PG -> five-part TPS -> SCI.

    Args:
        weights: Local paths for all six checkpoints; no implicit downloads.
        device: 'cpu' by default. Set 'cuda' or 'cuda:0' explicitly to use a GPU.
        sci_precision: 'float32' (default) or 'float16' on CUDA.
        cpu_offload: Move each torch model back to CPU after its stage.
        sampling_steps: KGI uses 50 respaced steps, with RePaint resampling.
        repaint_repeats: KGI uses 10. Set 1 for a faster smoke test.

    Models are loaded lazily, so preprocess() can be inspected before loading SCI.
    This is a reconstructed baseline, not a fit-aware/trained size model.
    """
    def __init__(self, weights: Weights, *, device="cpu", sci_precision="float32",
                 cpu_offload=True, sampling_steps=50, repaint_repeats=10, progress=True):
        weights.validate()
        if device != "cpu" and not (device == "cuda" or device.startswith("cuda:")):
            raise ValueError("device must be cpu, cuda or cuda:<index>")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable in this session")
        if sci_precision not in ("float32", "float16"):
            raise ValueError("sci_precision must be float32 or float16")
        if sci_precision == "float16" and device == "cpu":
            raise ValueError("Use float32 on CPU; float16 SCI requires an explicit CUDA device")
        if not isinstance(sampling_steps, int) or not 2 <= sampling_steps <= 1000:
            raise ValueError("sampling_steps must be an integer in [2,1000]")
        if not isinstance(repaint_repeats, int) or repaint_repeats < 1:
            raise ValueError("repaint_repeats must be a positive integer")
        self.weights, self.device = weights, torch.device(device)
        self.sci_precision, self.cpu_offload = sci_precision, cpu_offload
        self.sampling_steps, self.repaint_repeats = sampling_steps, repaint_repeats
        self.progress = progress
        self._garment = self._pose = self._parser = None
        self._kg = self._pg = self._sci = self._diffusion = self._conf = None

    @contextmanager
    def _on_device(self, model):
        model.to(self.device)
        try:
            yield model
        finally:
            if self.cpu_offload:
                model.to("cpu")

    @torch.no_grad()
    def preprocess(self, cloth_image, person_image, garment_type: str) -> PreparedInputs:
        garment_type = normalize_garment_type(garment_type)
        cloth, person = prepare_image(cloth_image), prepare_image(person_image)
        if self._garment is None:
            self._garment = GarmentLandmarks(self.weights.garment_hrnet)
            self._pose = HumanPose(self.weights.openpose_body)
            self._parser = HumanParser(self.weights.human_parser_lip)
        with self._on_device(self._garment.model):
            cloth_points, scores = self._garment.predict(cloth, garment_type, self.device)
        with self._on_device(self._pose.body.model):
            pose, missing = self._pose.predict(person, self.device)
        parse_lip, parse = self._parser.predict(person)
        agnostic, removal, skin = agnostic_conditions(parse, pose)
        notes = ["Preprocessing/masks are reconstructed; not an exact reproduction of author annotations.",
                 "Physical measurements are not used by this KGI baseline."]
        if garment_type == "short sleeve top":
            notes.append("Short-sleeve to 32-node mapping is an experimental contour extension.")
        if missing:
            notes.append(f"OpenPose fallback used for KGI joints {missing}.")
        metadata = {"garment_type": garment_type, "size_wh": [WIDTH, HEIGHT], "notes": notes,
                    "garment_landmark_scores": scores.tolist(), "pose_missing_joints": missing,
                    "parser": "SCHP LIP-20 ONNX", "pose_model": "OpenPose COCO-18 -> KGI-10"}
        return PreparedInputs(cloth, person, garment_type, cloth_points, pose, parse_lip,
                              parse, agnostic, removal, skin, metadata)

    def _load_kg_pg(self):
        from .vendor.kg import GCN_2
        from .vendor.pg import ParseGenerator
        if self._kg is None:
            self._kg = load_weights(GCN_2(adjacency(32, CONTOUR_EDGES + SYMMETRY_EDGES),
                                           adjacency(10, POSE_EDGES), 160), self.weights.kg)
        if self._pg is None:
            self._pg = load_weights(ParseGenerator(input_nc=19, output_nc=13, ngf=64), self.weights.pg)

    def _load_sci(self):
        if self._sci is not None:
            return
        from .vendor.sci.script_util import model_and_diffusion_defaults, create_model_and_diffusion
        options = model_and_diffusion_defaults()
        options.update(image_size=1024, num_classes=13, class_cond=True, learn_sigma=True,
                       num_channels=128, num_head_channels=32, num_heads=4, num_res_blocks=2,
                       attention_resolutions="32,16,8", resblock_updown=True, no_instance=True,
                       use_scale_shift_norm=True, use_checkpoint=False,
                       use_fp16=self.sci_precision == "float16", timestep_respacing=str(self.sampling_steps))
        self._conf = Config(options, inpa_inj_sched_prev=True, inpa_inj_sched_prev_cumnoise=False,
                            schedule_jump_params=dict(t_T=self.sampling_steps, n_sample=1,
                                                      jump_length=min(10, self.sampling_steps - 1),
                                                      jump_n_sample=self.repaint_repeats))
        model, self._diffusion = create_model_and_diffusion(**options, conf=self._conf)
        self._sci = load_weights(model, self.weights.sci)
        if self.sci_precision == "float16":
            self._sci.convert_to_fp16()

    @torch.no_grad()
    def __call__(self, cloth_image, person_image, garment_type: str, *, seed=42) -> TryOnResult:
        prepared = self.preprocess(cloth_image, person_image, garment_type)
        return self.infer_prepared(prepared, seed=seed)

    @torch.no_grad()
    def infer_prepared(self, prepared: PreparedInputs, *, seed=42) -> TryOnResult:
        started = time.perf_counter()
        self._load_kg_pg()
        center = (prepared.cloth_points[2] + prepared.cloth_points[18]) / 2
        cloth_points = torch.from_numpy(prepared.cloth_points - center)[None].to(self.device)
        pose = torch.from_numpy(prepared.pose)[None].to(self.device)
        with self._on_device(self._kg) as kg:
            predicted_points = kg(cloth_points, pose)[0].float().cpu().numpy()
        skeleton = draw_skeleton(torch.from_numpy(prepared.pose.copy()))
        contour = draw_cloth(torch.from_numpy(predicted_points.copy()))
        # Preserve the training contract: one-hot parse, normalized RGB sketches.
        pg_input = torch.cat((one_hot(prepared.parse_agnostic, self.device),
                              (skeleton[None].to(self.device) * 2 - 1),
                              (contour[None].to(self.device) * 2 - 1)), dim=1)
        with self._on_device(self._pg) as pg:
            predicted_parse = pg(pg_input).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        del pg_input, cloth_points, pose
        incomplete, keep, skipped = compose_conditions(prepared, predicted_points, predicted_parse)
        self._load_sci()
        kwargs = {"gt": image_tensor(incomplete, self.device),
                  "gt_keep_mask": torch.from_numpy(keep.astype(np.float32))[None, None].to(self.device),
                  "y": one_hot(predicted_parse, self.device)}
        cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] \
            if self.device.type == "cuda" else []
        # Scope RNG changes; a CPU invocation does not touch the CUDA generator.
        with torch.random.fork_rng(devices=cuda_devices), self._on_device(self._sci) as sci:
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(seed)

            def model_fn(x, t, y=None, **unused):
                return sci(x, t, y)

            sample = self._diffusion.p_sample_loop(
                model_fn, (1, 3, HEIGHT, WIDTH), model_kwargs=kwargs, device=self.device,
                progress=self.progress, return_all=False, clip_denoised=True, conf=self._conf)
            rgb = ((sample[0].float().cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        # Exact known-pixel preservation, including TPS texture, at the end of RePaint.
        rgb[keep] = incomplete[keep]
        metadata = dict(prepared.metadata, seed=seed, device=str(self.device),
                        sci_precision=self.sci_precision, sampling_steps=self.sampling_steps,
                        repaint_repeats=self.repaint_repeats, weights=self.weights.as_dict(),
                        skipped_tps_parts=skipped, inference_seconds=time.perf_counter() - started,
                        cloth_points=prepared.cloth_points.tolist(), pose=prepared.pose.tolist(),
                        predicted_points=predicted_points.tolist())
        artifacts = {"cloth": Image.fromarray(prepared.cloth), "person": Image.fromarray(prepared.person),
                     "cloth_landmarks": point_overlay(prepared.cloth, prepared.cloth_points),
                     "predicted_landmarks": point_overlay(prepared.person, predicted_points),
                     "parse_lip20": Image.fromarray(prepared.parse_lip),
                     "parse_kgi13": Image.fromarray(prepared.parse),
                     "parse_agnostic": Image.fromarray(prepared.parse_agnostic),
                     "parse_predicted": Image.fromarray(predicted_parse),
                     "removal_mask": Image.fromarray(prepared.removal.astype(np.uint8) * 255),
                     "keep_mask": Image.fromarray(keep.astype(np.uint8) * 255),
                     "incomplete": Image.fromarray(incomplete)}
        return TryOnResult(Image.fromarray(rgb), artifacts, metadata)
