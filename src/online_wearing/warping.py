"""Five-part KGI TPS, using a CPU inverse mapping and a modern linear solve."""
import cv2
import numpy as np

from .contracts import TPS_GROUPS


def inverse_tps(source, target, height, width, regularization=1e-6):
    """Return pixel sampling maps: target grid -> source. No CUDA allocations."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    unique = []
    for i in range(len(source)):
        if all(np.linalg.norm(source[i] - source[j]) > 1e-6 and
               np.linalg.norm(target[i] - target[j]) > 1e-6 for j in unique):
            unique.append(i)
    source, target = source[unique], target[unique]
    if len(source) < 3 or any(np.linalg.matrix_rank(p - p.mean(axis=0), tol=1e-6) < 2
                              for p in (source, target)):
        raise ValueError("Degenerate TPS part (fewer than three noncollinear correspondences)")
    n = len(source)
    distances = np.square(target[:, None] - target[None]).sum(axis=-1)
    kernel = distances * np.log(distances + 1e-9)
    kernel += np.eye(n) * regularization
    polynomial = np.column_stack((np.ones(n), target))
    system = np.block([[kernel, polynomial], [polynomial.T, np.zeros((3, 3))]])
    coefficients = np.linalg.solve(system, np.vstack((source, np.zeros((3, 2)))))
    maps = np.empty((height * width, 2), dtype=np.float32)
    # Bounded temporary memory: never allocate H×W×K×2 at 1024 resolution.
    for start in range(0, len(maps), 32768):
        indices = np.arange(start, min(start + 32768, len(maps)))
        grid = np.column_stack(((indices % width) / width, (indices // width) / height))
        distances = np.square(grid[:, None] - target[None]).sum(axis=-1)
        radial = distances * np.log(distances + 1e-9)
        mapped = radial @ coefficients[:n] + np.column_stack((np.ones(len(grid)), grid)) @ coefficients[n:]
        maps[start:start + len(grid)] = mapped * [width, height]
    if not np.isfinite(maps).all():
        raise ValueError("Nonfinite TPS mapping; inspect garment landmarks")
    maps = maps.reshape(height, width, 2)
    return maps[:, :, 0].copy(), maps[:, :, 1].copy()


def polygon_mask(points, shape):
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    polygon = np.rint(points * [width, height]).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def compose_conditions(prepared, predicted_points, predicted_parse):
    """Return incomplete RGB, keep mask, and skipped degenerate sleeve parts."""
    parse, image = prepared.parse, prepared.person
    clothing = predicted_parse == 3
    if np.count_nonzero(clothing) < 64:
        raise ValueError("PG did not predict an upper-garment region")
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    removal = prepared.removal | (cv2.dilate(clothing.astype(np.uint8), grow) > 0) | (parse == 3)
    # Reconstructed semantic agreement refinement from the supplementary description.
    agreed = (parse == predicted_parse) & ~np.isin(parse, [0, 3])
    removal[agreed] = False
    removal[np.isin(parse, [1, 2])] = False  # hair/face always preserved
    keep = ~removal
    skin = prepared.skin & np.isin(predicted_parse, [5, 6]) & ~clothing
    keep |= skin
    incomplete = image.copy()
    incomplete[~keep] = 0
    cloth_covered = np.zeros(parse.shape, dtype=bool)
    skipped = []
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for part, group in enumerate(TPS_GROUPS):
        source, target = prepared.cloth_points[list(group)], predicted_points[list(group)]
        try:
            map_x, map_y = inverse_tps(source, target, *parse.shape)
        except (ValueError, np.linalg.LinAlgError) as error:
            if part == 0:
                raise ValueError("TPS torso failed; inspect the landmark overlay") from error
            skipped.append(part)
            continue  # Sleeveless and short-sleeve distal parts can legitimately collapse.
        source_mask = polygon_mask(source, parse.shape)
        target_mask = polygon_mask(target, parse.shape)
        warped_mask = cv2.remap(source_mask, map_x, map_y, cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        support = (warped_mask > 0) & (target_mask > 0)
        support = cv2.erode(support.astype(np.uint8), erode) > 0
        support &= clothing & ~np.isin(parse, [1, 2])
        warped = cv2.remap(prepared.cloth, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        incomplete[support] = warped[support]
        keep[support] = True
        cloth_covered |= support
    if np.count_nonzero(cloth_covered) < 64:
        raise ValueError("TPS produced almost no valid cloth; inspect the contour/type")
    return incomplete, keep, skipped
