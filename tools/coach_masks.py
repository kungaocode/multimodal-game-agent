"""Pixel-level deploy masking and landing-point planning.

These functions compute the "no land" mask (red-zone closed area + bottom UI
band + screen edge guard) from a screenshot, then snap a deploy landing point
to the nearest landable grass. They are pure image-processing utilities with
no dependency on FSM or ADB.
"""

from __future__ import annotations

import math

from PIL import Image


def red_zone_mask(screenshot: Image.Image):
    """Per-pixel enemy red-zone mask: translucent red overlay + strong red UI.

    Translucent red zone (pinkish overlay): R clearly above G/B but not
    oversaturated; strong red: red buildings/buttons. The bottom UI band
    (troop bar, battle buttons) is also captured; the snapping functions
    additionally exclude the UI height band.
    """
    import numpy as np

    a = np.asarray(screenshot.convert("RGB")).astype(int)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    overlay = (R > G + 8) & (R > B + 8) & (R > 130) & (R < 250) & (G > 80)
    strong = (R > 160) & (R - G > 60) & (R - B > 60)
    return overlay | strong


def green_mask(screenshot: Image.Image):
    """Per-pixel grass mask: bright green (G clearly dominant and not dark)."""
    import numpy as np

    a = np.asarray(screenshot.convert("RGB")).astype(int)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    return (G > R + 18) & (G > B + 10) & (G > 100)


def no_land_mask(
    screenshot: Image.Image,
    ui_ratio: float = 0.86,
    dilate: int = 6,
    edge_guard: int = 3,
):
    """No-land mask (True = forbidden deploy point).

    The red zone in the game forms an enclosed area: not only the red pixels
    themselves, but everything inside the red boundary is also no-land. A
    flood fill from the screen border finds the outside; unreached non-red
    pixels are the enclosed interior. A small dilation closes boundary gaps.
    The bottom UI band and screen edge guard are appended last.
    """
    import numpy as np
    from scipy import ndimage

    red = red_zone_mask(screenshot)
    h, w = red.shape

    border = np.zeros_like(red)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    outside = ndimage.binary_propagation(border & ~red, mask=~red)
    no_land = red | (~outside)

    if dilate > 0:
        no_land = ndimage.binary_dilation(no_land, iterations=dilate)

    ui_y = int(h * ui_ratio)
    no_land[ui_y:, :] = True
    if edge_guard > 0:
        no_land[:edge_guard, :] = True
        no_land[-edge_guard:, :] = True
        no_land[:, :edge_guard] = True
        no_land[:, -edge_guard:] = True
    return no_land


def snap_deploy_point(
    coords: tuple[int, int],
    screenshot: Image.Image | None,
    margin: int = 40,
    max_radius: int = 300,
    ui_ratio: float = 0.86,
) -> tuple[int, int]:
    """Snap a deploy landing point: never land on a no-land area.

    Strategy:
      1) Already on landable grass -> return as-is;
      2) Otherwise find the nearest grass within max_radius (distance
         transform), staying close to the target building;
      3) No grass nearby -> relax to any non-no-land point (2x radius);
      4) Still nothing -> return the original coords (trust the coach).
    """
    if screenshot is None:
        return coords
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError:
        return coords
    w, h = screenshot.size
    x = min(max(int(coords[0]), 0), w - 1)
    y = min(max(int(coords[1]), 0), h - 1)
    no_land = no_land_mask(screenshot, ui_ratio=ui_ratio)
    green = green_mask(screenshot)

    def nearest(allowed: np.ndarray, cap: float) -> tuple[int, int] | None:
        if not allowed.any():
            return None
        dist, idx = ndimage.distance_transform_edt(~allowed, return_indices=True)
        d = float(dist[y, x])
        if 0 < d <= cap:
            return int(idx[1, y, x]), int(idx[0, y, x])
        return None

    if not no_land[y, x] and green[y, x]:
        return (x, y)
    allowed = green & ~no_land
    pt = nearest(allowed, float(max_radius))
    if pt is not None:
        return pt
    allowed = ~no_land
    pt = nearest(allowed, float(max_radius * 2))
    if pt is not None:
        return pt
    return (x, y)


def plan_deploy_point(
    coords: tuple[int, int],
    target_name: str | None,
    detections: list[dict] | None,
    screenshot: Image.Image | None,
    max_radius: int = 300,
    tight_px: int = 120,
) -> tuple[int, int]:
    """Plan a deploy landing point adjacent to the target resource building.

    Core logic (exclude no-land first, then pick the nearest point to the
    target building):
      1) Use no_land_mask for "red-zone closed area + bottom UI" mask;
      2) Target center: prefer the detection box matching target_name (when
         multiple instances exist, pick the one nearest to the coach coords),
         else fall back to the coach coords;
      3) Find the nearest landable point near the target center (grass +
         outside red zone + above UI band) so the landing is always adjacent
         to the building and never inside the red zone;
      4) If the coach's own landing point is already landable and close to
         the target center (<= tight_px), trust it directly;
      5) On any failure, fall back to snap_deploy_point which guarantees the
         point is never inside the red zone.
    """
    if screenshot is None:
        return coords
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError:
        return coords
    w, h = screenshot.size
    no_land = no_land_mask(screenshot)
    green = green_mask(screenshot)
    land_grass = green & ~no_land
    land_any = ~no_land

    def clamp(pt: tuple[int, int]) -> tuple[int, int]:
        return min(max(int(pt[0]), 0), w - 1), min(max(int(pt[1]), 0), h - 1)

    def nearest_to(center: tuple[int, int], land: object, cap: float) -> tuple[int, int] | None:
        if not land.any():
            return None
        dist, idx = ndimage.distance_transform_edt(~land, return_indices=True)
        d = float(dist[center[1], center[0]])
        if d == 0:
            return center
        if d <= cap:
            return (int(idx[1, center[1], center[0]]), int(idx[0, center[1], center[0]]))
        return None

    target_center: tuple[int, int] | None = None
    if detections and target_name:
        cands = [
            tuple(d["coords"])
            for d in detections
            if d.get("type") == target_name and d.get("coords")
        ]
        if cands:
            target_center = min(cands, key=lambda c: math.dist(c, coords))
    if target_center is None:
        target_center = coords
    cx, cy = clamp(target_center)

    if not no_land[cy, cx] and green[cy, cx]:
        nearest_pt = (cx, cy)
    else:
        nearest_pt = nearest_to((cx, cy), land_grass, float(max_radius))
        if nearest_pt is None:
            nearest_pt = nearest_to((cx, cy), land_any, float(max_radius))

    ox, oy = clamp(coords)
    if not no_land[oy, ox] and math.dist((ox, oy), target_center) <= tight_px:
        return (ox, oy)

    if nearest_pt is not None:
        return nearest_pt
    return snap_deploy_point(coords, screenshot)
