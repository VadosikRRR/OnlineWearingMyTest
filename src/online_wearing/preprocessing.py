"""Local-weight adapters for GarmentIQ HRNet, OpenPose COCO-18 and SCHP LIP-20."""
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from .checkpoints import load_weights
from .contracts import (WIDTH, HEIGHT, COCO18_TO_KGI, LANDMARK_MAPS,
                        LANDMARK_COUNTS, LIP_TO_KGI)


def read_image(value) -> Image.Image:
    """Paths/PIL/uint8 RGB arrays. Composite transparent product images on white."""
    if isinstance(value, np.ndarray):
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] not in (3, 4):
            raise ValueError("Array input must be uint8 H×W×3 RGB or H×W×4 RGBA")
        im = Image.fromarray(value)
    elif isinstance(value, Image.Image):
        im = value.copy()
    else:
        with Image.open(value) as opened:
            im = opened.copy()
    im = ImageOps.exif_transpose(im)
    if "A" in im.getbands() or "transparency" in im.info:
        rgba = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), rgba)
    return im.convert("RGB")


def prepare_image(value) -> np.ndarray:
    # Single documented resize, before all detectors. No mismatched crop/coordinates.
    return np.array(read_image(value).resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS))


def landmarks_to_kgi(coords, garment_type):
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape != (LANDMARK_COUNTS[garment_type], 2) or not np.isfinite(coords).all():
        raise ValueError(f"Invalid {garment_type} landmark array: {coords.shape}")
    points = coords[list(LANDMARK_MAPS[garment_type])] / np.array([WIDTH, HEIGHT])
    if (points < -0.1).any() or (points > 1.1).any():
        raise ValueError("Garment landmarks are outside the prepared image; check the input/type")
    return np.clip(points, 0, 1).astype(np.float32)


def lip_to_kgi(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("LIP parse must be a 2D integer label map")
    if labels.min() < 0 or labels.max() > 19:
        raise ValueError("Expected LIP-20 class IDs 0..19")
    return np.asarray(LIP_TO_KGI, dtype=np.uint8)[labels]


class GarmentLandmarks:
    def __init__(self, weight_path):
        from garmentiq.landmark.detection.model_definition import PoseHighResolutionNet
        # Same architecture/checkpoint as needed.py, with strict loading and explicit device.
        # Avoid library load_model's version-dependent automatic device / strict=False.
        self.model = load_weights(PoseHighResolutionNet(), weight_path)

    @torch.no_grad()
    def predict(self, image, garment_type, device):
        from garmentiq.garment_classes import garment_classes
        from garmentiq.landmark.detection.utils import (
            input_image_transform, get_final_preds, transform_preds,
        )
        tensor, _, center, scale = input_image_transform(image)
        self.model.to(device)
        start, end = garment_classes[garment_type]["index_range"]
        heatmaps = self.model(tensor.to(device))[:, start:end].float().cpu().numpy()
        coords, scores = get_final_preds(heatmaps, height=heatmaps.shape[2], width=heatmaps.shape[3])
        coords = transform_preds(coords[0], center, scale,
                                 output_size=[heatmaps.shape[3], heatmaps.shape[2]])
        selected = scores[0, list(LANDMARK_MAPS[garment_type]), 0]
        if not np.isfinite(selected).all() or np.count_nonzero(selected > 0) < 24:
            raise ValueError("HRNet could not locate enough garment landmarks")
        return landmarks_to_kgi(coords, garment_type), selected


class HumanPose:
    def __init__(self, weight_path):
        from controlnet_aux.open_pose.body import Body
        from controlnet_aux.open_pose.model import bodypose_model
        from controlnet_aux.open_pose.util import transfer
        # Body's original loader lacks map_location. Preserve its name conversion,
        # but load CPU tensors explicitly and do not load unused face/hand models.
        self.body = Body.__new__(Body)
        model = bodypose_model()
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(transfer(model, state), strict=True)
        self.body.model = model.eval()

    @torch.no_grad()
    def predict(self, image, device):
        self.body.to(device)
        candidates, subsets = self.body(image[:, :, ::-1].copy())  # Body expects BGR.
        bodies = self.body.format_body_result(candidates, subsets)
        if len(bodies) != 1:
            raise ValueError(f"Expected one person, OpenPose found {len(bodies)}; use a single-person photo")
        keypoints = bodies[0].keypoints
        required = (1, 2, 5, 8, 11)  # neck, shoulders, hips: essential torso anchors
        if any(keypoints[i] is None for i in required):
            raise ValueError("OpenPose missed neck/shoulders/hips; use a visible upper body and hips")
        points = np.zeros((10, 2), dtype=np.float32)
        missing = []
        for out_index, in_index in enumerate(COCO18_TO_KGI):
            point = keypoints[in_index]
            if point is None:
                missing.append(out_index)
                points[out_index] = points[1] if out_index == 0 else points[out_index - 1]
            else:
                points[out_index] = (point.x / WIDTH, point.y / HEIGHT)
        # Nose appears before neck in KGI order.
        if 0 in missing:
            points[0] = points[1]
        return np.clip(points, 0, 1), missing


class HumanParser:
    """SCHP LIP-20 ONNX export from yisol/IDM-VTON. ORT runs explicitly on CPU."""
    def __init__(self, weight_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(weight_path), providers=["CPUExecutionProvider"])

    def predict(self, rgb):
        h, w = rgb.shape[:2]
        # Match SCHP SimpleFolderDataset: square padded affine of [0,0,w-1,h-1].
        side = float(max(w - 1, h - 1))
        center = np.array([(w - 1) / 2, (h - 1) / 2], dtype=np.float32)
        factor = 511.0 / side
        affine = np.array([[factor, 0, 255.5 - center[0] * factor],
                           [0, factor, 255.5 - center[1] * factor]], dtype=np.float32)
        bgr = cv2.warpAffine(rgb[:, :, ::-1], affine, (512, 512), flags=cv2.INTER_LINEAR)
        normalized = (bgr.astype(np.float32) / 255 - np.array([.406, .456, .485], np.float32))
        normalized /= np.array([.225, .224, .229], np.float32)
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: tensor})
        candidates = []
        for output in outputs:
            output = np.asarray(output)
            while output.ndim > 4 and output.shape[0] == 1:
                output = output[0]
            if output.ndim == 4 and output.shape[:2] == (1, 20):
                candidates.append(output)
        if not candidates:
            raise ValueError("Wrong ONNX parser: expected SCHP LIP output [1,20,H,W], not ATR-18")
        logits = F.interpolate(torch.from_numpy(candidates[-1]), size=(512, 512),
                               mode="bilinear", align_corners=True)[0].numpy()
        inverse = cv2.invertAffineTransform(affine)
        restored = np.stack([cv2.warpAffine(channel, inverse, (w, h), flags=cv2.INTER_LINEAR)
                             for channel in logits], axis=0)
        labels = restored.argmax(axis=0).astype(np.uint8)
        return labels, lip_to_kgi(labels)


def agnostic_conditions(parse, pose):
    """Reconstructed upper-body removal; not the unpublished KGI preprocessing."""
    height, width = parse.shape
    xy = np.rint(pose * [width, height]).astype(np.int32)
    shoulder_width = np.linalg.norm(xy[2] - xy[5])
    radius = max(8, round(shoulder_width * 0.12))
    coarse = np.zeros_like(parse, dtype=np.uint8)
    torso = xy[[2, 5, 9, 8]]
    cv2.fillConvexPoly(coarse, cv2.convexHull(torso), 255)
    for chain in ((2, 3, 4), (5, 6, 7)):
        cv2.polylines(coarse, [xy[list(chain)]], False, 255, radius * 2)
        for joint in chain:
            cv2.circle(coarse, tuple(xy[joint]), radius, 255, -1)
    coarse[parse == 3] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    coarse = cv2.dilate(coarse, kernel)
    # Preserve identity, lower-body clothes, legs/shoes/socks and accessories.
    coarse[np.isin(parse, [1, 2, 4, 7, 8, 9, 10, 11, 12])] = 0
    agnostic = parse.copy()
    agnostic[(coarse > 0) | np.isin(parse, [3, 5, 6])] = 0
    skin = np.isin(parse, [5, 6])
    return agnostic, coarse > 0, skin


@dataclass
class PreparedInputs:
    cloth: np.ndarray
    person: np.ndarray
    garment_type: str
    cloth_points: np.ndarray
    pose: np.ndarray
    parse_lip: np.ndarray
    parse: np.ndarray
    parse_agnostic: np.ndarray
    removal: np.ndarray
    skin: np.ndarray
    metadata: dict
