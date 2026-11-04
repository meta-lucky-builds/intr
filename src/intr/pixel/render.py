"""Batched rasterizer in pure torch. A pure function of the state and a palette: same input, same image.

Hidden physics never enters. The rod is drawn at a fixed length whatever the pendulum's true length, and masses, friction, inertia,
drag, wind and actuator changes draw nothing, so they can only show through motion. render takes no environment, which is what
guarantees this, and a test checks it with different hidden values.

Soft edges: coverage = clamp((r - d) / pixel + 0.5, 0, 1), one pixel wide, so sub-pixel motion changes the image.

Canvas half-widths. Puck stays in its box. The drone can fly far outside the task region under random actions, and a canvas covering
all of that would make the task region a few pixels wide. So its drawn position is clamped at the canvas edge. Attitude stays visible,
position saturates there, and offscreen_fraction reports how often this happens, per tier.
"""

from __future__ import annotations

import torch
from torch import Tensor

STATE_DIMS = {"pendulum": 2, "puck": 8, "drone": 6}
CANVAS = {"pendulum": 1.25, "puck": 1.15, "drone": 3.5}
DRONE_DRAW_LIMIT = 2.8
ROD_LENGTH, ROD_HALF_WIDTH, BOB_RADIUS = 1.0, 0.05, 0.16
PUCK_RADIUS = 0.1  # PuckEnv.radius: contact is visible in the image, and it is not hidden
DRONE_ARM, DRONE_BODY_HALF_WIDTH, DRONE_ROTOR_RADIUS, DRONE_NOSE = 0.6, 0.12, 0.22, 0.3

# background first, then one colour per layer
_PALETTES = {
    "pendulum": [(0.08, 0.08, 0.10), (0.85, 0.85, 0.85), (0.95, 0.35, 0.25)],
    "puck": [(0.08, 0.08, 0.10), (0.30, 0.60, 0.95), (0.95, 0.70, 0.20)],
    "drone": [(0.08, 0.08, 0.10), (0.85, 0.85, 0.85), (0.95, 0.35, 0.25), (0.30, 0.85, 0.40), (0.60, 0.40, 0.90)],
}


def train_palette(family: str) -> Tensor:
    """(L + 1, 3), used for every tier but appearance."""
    return torch.tensor(_PALETTES[family], dtype=torch.float32)


def appearance_palette(family: str, n_envs: int, gen: torch.Generator) -> Tensor:
    """(E, L + 1, 3). New colours and background per environment, keeping dark background and bright objects (H16).

    Contrast polarity stays as in training, so a failure is not guaranteed by construction.
    """
    layers = len(_PALETTES[family]) - 1
    background = 0.30 * torch.rand(n_envs, 1, 3, generator=gen)
    foreground = 0.50 + 0.50 * torch.rand(n_envs, layers, 3, generator=gen)
    return torch.cat([background, foreground], dim=1)


def _grid(size: int, half: float, device) -> tuple[Tensor, Tensor]:
    """Pixel centers, (1, H, W) each. Row 0 is the top of the picture."""
    step = 2.0 * half / size
    coords = -half + step * (torch.arange(size, device=device, dtype=torch.float32) + 0.5)
    y, x = torch.meshgrid(coords.flip(0), coords, indexing="ij")
    return x[None], y[None]


def _disc(x: Tensor, y: Tensor, cx: Tensor, cy: Tensor, radius: float, pixel: float) -> Tensor:
    """cx, cy (N,). Returns coverage (N, H, W)."""
    d = torch.sqrt((x - cx[:, None, None]) ** 2 + (y - cy[:, None, None]) ** 2 + 1e-12)
    return ((radius - d) / pixel + 0.5).clamp(0.0, 1.0)


def _capsule(x: Tensor, y: Tensor, ax: Tensor, ay: Tensor, bx: Tensor, by: Tensor, radius: float, pixel: float) -> Tensor:
    """Segment from a to b with rounded ends. Inputs (N,). Returns coverage (N, H, W)."""
    ax, ay = ax[:, None, None], ay[:, None, None]
    dx, dy = bx[:, None, None] - ax, by[:, None, None] - ay
    t = (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy + 1e-12)).clamp(0.0, 1.0)
    d = torch.sqrt((x - ax - t * dx) ** 2 + (y - ay - t * dy) ** 2 + 1e-12)
    return ((radius - d) / pixel + 0.5).clamp(0.0, 1.0)


def _layers(family: str, s: Tensor, x: Tensor, y: Tensor, pixel: float) -> list[Tensor]:
    """Coverage per layer, painter's order. s (N, S)."""
    if family == "pendulum":
        theta = s[:, 0]  # 0 is upright
        tip_x, tip_y = ROD_LENGTH * torch.sin(theta), ROD_LENGTH * torch.cos(theta)
        zero = torch.zeros_like(theta)
        return [_capsule(x, y, zero, zero, tip_x, tip_y, ROD_HALF_WIDTH, pixel), _disc(x, y, tip_x, tip_y, BOB_RADIUS, pixel)]
    if family == "puck":
        return [_disc(x, y, s[:, 0], s[:, 1], PUCK_RADIUS, pixel), _disc(x, y, s[:, 2], s[:, 3], PUCK_RADIUS, pixel)]
    if family == "drone":
        cx, cy = s[:, 0].clamp(-DRONE_DRAW_LIMIT, DRONE_DRAW_LIMIT), s[:, 1].clamp(-DRONE_DRAW_LIMIT, DRONE_DRAW_LIMIT)
        c, si = torch.cos(s[:, 2]), torch.sin(s[:, 2])
        left_x, left_y = cx - DRONE_ARM * c, cy - DRONE_ARM * si
        right_x, right_y = cx + DRONE_ARM * c, cy + DRONE_ARM * si
        nose_x, nose_y = cx - DRONE_NOSE * si, cy + DRONE_NOSE * c  # along thrust
        return [
            _capsule(x, y, left_x, left_y, right_x, right_y, DRONE_BODY_HALF_WIDTH, pixel),
            _disc(x, y, left_x, left_y, DRONE_ROTOR_RADIUS, pixel),
            _disc(x, y, right_x, right_y, DRONE_ROTOR_RADIUS, pixel),
            _capsule(x, y, cx, cy, nose_x, nose_y, DRONE_BODY_HALF_WIDTH, pixel),
        ]
    raise ValueError(f"unknown family {family!r}")


@torch.no_grad()
def render(family: str, state: Tensor, size: int, palette: Tensor | None = None, chunk: int = 2048) -> Tensor:
    """state (..., S) -> uint8 (..., 3, H, W), on the state's device.

    palette: (L + 1, 3) for every image, or (E, L + 1, 3) with E the first dimension of state.
    """
    device = state.device
    shape = state.shape[:-1]
    flat = state.reshape(-1, state.shape[-1]).to(torch.float32)
    palette = train_palette(family) if palette is None else palette
    palette = palette.to(device).to(torch.float32)
    if palette.dim() == 3:
        view = (palette.shape[0],) + (1,) * (len(shape) - 1) + palette.shape[1:]
        colours = palette.reshape(view).expand(*shape, *palette.shape[1:]).reshape(-1, *palette.shape[1:])
    else:
        colours = palette[None].expand(flat.shape[0], -1, -1)
    half = CANVAS[family]
    pixel = 2.0 * half / size
    x, y = _grid(size, half, device)
    parts = []
    for start in range(0, flat.shape[0], chunk):
        s = flat[start : start + chunk]
        c = colours[start : start + chunk]  # (N, L + 1, 3)
        img = c[:, 0, :, None, None].expand(-1, -1, size, size)
        for i, coverage in enumerate(_layers(family, s, x, y, pixel)):
            alpha = coverage[:, None]  # (N, 1, H, W)
            img = img * (1.0 - alpha) + c[:, i + 1, :, None, None] * alpha
        parts.append((img * 255.0).round().clamp(0.0, 255.0).to(torch.uint8))
    return torch.cat(parts, dim=0).reshape(*shape, 3, size, size)


def offscreen_fraction(family: str, state: Tensor) -> float:
    """Share of frames whose drawn position was clamped. Zero except on the drone."""
    if family != "drone":
        return 0.0
    return (state[..., :2].abs() > DRONE_DRAW_LIMIT).any(dim=-1).float().mean().item()


def stack_windows(frames: Tensor) -> Tensor:
    """(N, stack, 3, H, W) uint8 -> (N, 3 stack, H, W) float32 in [0, 1], oldest frame first."""
    return frames.flatten(1, 2).to(torch.float32) / 255.0
