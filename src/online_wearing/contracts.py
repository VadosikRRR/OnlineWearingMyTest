"""Dependency-free, explicit model and annotation contracts."""
from dataclasses import dataclass, fields
from pathlib import Path

WIDTH, HEIGHT = 768, 1024

# These are ZERO-BASED DeepFashion2 class-local indices, not arbitrary contour samples.
LONG_TO_KGI = tuple(range(1, 33))
VEST_TO_KGI = (1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7,
               8, 9, 10, 11, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14)
# Experimental extension: collapse the distal long-sleeve nodes onto the short cuff.
SHORT_TO_KGI = (1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 9, 10, 10, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 20, 20, 21, 22, 22, 22, 23, 24)
LANDMARK_MAPS = {"long sleeve top": LONG_TO_KGI, "short sleeve top": SHORT_TO_KGI,
                 "vest": VEST_TO_KGI}
LANDMARK_COUNTS = {"long sleeve top": 33, "short sleeve top": 25, "vest": 15}
ALIASES = {"long_sleeve_top": "long sleeve top", "short_sleeve_top": "short sleeve top",
           "t-shirt": "short sleeve top", "tshirt": "short sleeve top",
           "футболка": "short sleeve top", "лонгслив": "long sleeve top",
           "майка": "vest", "sleeveless": "vest"}

# BODY_25 -> KGI and COCO-18 -> KGI have identical named joints, different hip indices.
BODY25_TO_KGI = (0, 1, 2, 3, 4, 5, 6, 7, 9, 12)
COCO18_TO_KGI = (0, 1, 2, 3, 4, 5, 6, 7, 8, 11)
CONTOUR_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 31)) + tuple(
    (i, i - 1) for i in range(31, 5, -1)) + ((5, 0),)
SYMMETRY_EDGES = ((0, 4), (1, 3), (5, 31), (14, 22), (15, 21), (16, 20),
                  (17, 19), (6, 13), (7, 12), (8, 11), (23, 30), (24, 29),
                  (25, 28), (2, 18))
POSE_EDGES = ((0, 1), (1, 2), (1, 5), (2, 3), (5, 6), (3, 4), (6, 7), (1, 8), (1, 9))
TPS_GROUPS = ((4, 3, 2, 1, 0, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 31),
              (5, 6, 7, 12, 13, 14), (7, 8, 9, 10, 11, 12),
              (22, 23, 24, 29, 30, 31), (24, 25, 26, 27, 28, 29))

# LIP-20 -> HR-VITON/KGI-13; channel 11 is socks, not neck/skin.
LIP_TO_KGI = (0, 1, 1, 12, 2, 3, 3, 3, 11, 4, 0, 12, 4, 2, 5, 6, 7, 8, 9, 10)
MEASUREMENT_FIELDS = ("body_bust", "body_height", "body_hips", "body_waist",
                      "garment_bust", "garment_length", "garment_sleeve_length")


def normalize_garment_type(value: str) -> str:
    key = value.strip().lower()
    key = ALIASES.get(key, key)
    if key not in LANDMARK_MAPS:
        raise ValueError(f"Unsupported garment type {value!r}. KGI supports upper-body garments; "
                         f"choose {tuple(LANDMARK_MAPS)}. Dresses/bottoms need retraining.")
    return key


@dataclass(frozen=True)
class Weights:
    """All six weight files are local; construction never downloads anything."""
    kg: str | Path
    pg: str | Path
    sci: str | Path
    garment_hrnet: str | Path
    openpose_body: str | Path
    human_parser_lip: str | Path

    def validate(self) -> None:
        missing = [f"{f.name}: {getattr(self, f.name)}" for f in fields(self)
                   if not Path(getattr(self, f.name)).is_file()]
        if missing:
            raise FileNotFoundError("Missing weights:\n" + "\n".join(missing))

    def as_dict(self) -> dict[str, str]:
        return {f.name: str(Path(getattr(self, f.name)).resolve()) for f in fields(self)}
