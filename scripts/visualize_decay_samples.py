from __future__ import annotations

import math
import random
from pathlib import Path

import artifact_paths as ap


WIDTH = 1180
HEIGHT = 820
MARGIN = 64
PANEL_GAP = 58
OUTFILE = ap.DECAY_THETA_EXAMPLES


def model(t: float, amplitude: float, decay_rate: float) -> float:
    return amplitude * math.exp(-decay_rate * t)


def simulate_observation(
    *,
    amplitude: float,
    decay_rate: float,
    noise: float,
    times: list[float],
    rng: random.Random,
) -> list[tuple[float, float]]:
    return [
        (t, model(t, amplitude, decay_rate) + rng.gauss(0.0, noise))
        for t in times
    ]


def sx(t: float, xmin: float, xmax: float, left: float, panel_width: float) -> float:
    return left + (t - xmin) / (xmax - xmin) * panel_width


def sy(y: float, ymin: float, ymax: float, top: float, panel_height: float) -> float:
    return top + (ymax - y) / (ymax - ymin) * panel_height


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 16,
    weight: int = 400,
    anchor: str = "start",
    color: str = "#172033",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{color}">{text}</text>'
    )


def make_panel(
    *,
    left: float,
    top: float,
    panel_width: float,
    panel_height: float,
    title: str,
    theta_label: str,
    times: list[float],
    points: list[tuple[float, float]],
    amplitude: float,
    decay_rate: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    color: str,
) -> list[str]:
    parts: list[str] = []
    right = left + panel_width
    bottom = top + panel_height

    parts.append(
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{panel_width:.1f}" '
        f'height="{panel_height:.1f}" rx="8" fill="#fbfcff" stroke="#d8dee9"/>'
    )

    for frac in [0.25, 0.5, 0.75]:
        x = left + frac * panel_width
        y = top + frac * panel_height
        parts.append(
            f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
            f'stroke="#edf1f6" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" '
            f'stroke="#edf1f6" stroke-width="1"/>'
        )

    parts.append(
        f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" '
        f'stroke="#586174" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" '
        f'stroke="#586174" stroke-width="1.2"/>'
    )

    curve_points = []
    for i in range(120):
        t = xmin + (xmax - xmin) * i / 119
        y = model(t, amplitude, decay_rate)
        curve_points.append(
            f"{sx(t, xmin, xmax, left, panel_width):.1f},"
            f"{sy(y, ymin, ymax, top, panel_height):.1f}"
        )
    parts.append(
        f'<polyline points="{" ".join(curve_points)}" fill="none" '
        f'stroke="{color}" stroke-width="3.2" stroke-linecap="round"/>'
    )

    for t, y in points:
        parts.append(
            f'<circle cx="{sx(t, xmin, xmax, left, panel_width):.1f}" '
            f'cy="{sy(y, ymin, ymax, top, panel_height):.1f}" r="4.2" '
            f'fill="{color}" fill-opacity="0.74" stroke="#ffffff" stroke-width="1"/>'
        )

    parts.append(svg_text(left, top - 18, title, size=19, weight=700))
    parts.append(svg_text(left, top + 22, theta_label, size=14, color="#475066"))
    parts.append(svg_text(left, bottom + 30, "time t", size=13, color="#475066"))
    parts.append(
        svg_text(left - 18, top - 4, "y", size=13, anchor="end", color="#475066")
    )
    parts.append(
        svg_text(left, bottom + 16, f"{xmin:g}", size=12, color="#667085")
    )
    parts.append(
        svg_text(right, bottom + 16, f"{xmax:g}", size=12, anchor="end", color="#667085")
    )
    parts.append(
        svg_text(left - 8, bottom + 4, f"{ymin:g}", size=12, anchor="end", color="#667085")
    )
    parts.append(
        svg_text(left - 8, top + 4, f"{ymax:g}", size=12, anchor="end", color="#667085")
    )
    return parts


def main() -> None:
    rng = random.Random(11)
    times = [6.0 * i / 31 for i in range(32)]
    scenarios = [
        {
            "title": "same amplitude, slow decay, low noise",
            "theta": (5.0, 0.35, 0.20),
            "color": "#2f6fbb",
        },
        {
            "title": "same amplitude, faster decay",
            "theta": (5.0, 0.95, 0.20),
            "color": "#bf4d5a",
        },
        {
            "title": "larger amplitude, same decay",
            "theta": (8.0, 0.35, 0.20),
            "color": "#3d7f4b",
        },
        {
            "title": "same curve, higher observation noise",
            "theta": (5.0, 0.35, 0.85),
            "color": "#8a5fbf",
        },
    ]

    simulated = []
    all_y = []
    for scenario in scenarios:
        amplitude, decay_rate, noise = scenario["theta"]
        points = simulate_observation(
            amplitude=amplitude,
            decay_rate=decay_rate,
            noise=noise,
            times=times,
            rng=rng,
        )
        simulated.append((scenario, points))
        all_y.extend(y for _, y in points)
        all_y.extend(model(t, amplitude, decay_rate) for t in times)

    ymin = math.floor(min(all_y) - 0.4)
    ymax = math.ceil(max(all_y) + 0.4)
    xmin = min(times)
    xmax = max(times)

    panel_width = (WIDTH - 2 * MARGIN - PANEL_GAP) / 2
    panel_height = (HEIGHT - 2 * MARGIN - PANEL_GAP - 62) / 2
    panel_positions = [
        (MARGIN, MARGIN + 68),
        (MARGIN + panel_width + PANEL_GAP, MARGIN + 68),
        (MARGIN, MARGIN + 68 + panel_height + PANEL_GAP),
        (MARGIN + panel_width + PANEL_GAP, MARGIN + 68 + panel_height + PANEL_GAP),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(
            MARGIN,
            42,
            "Observed datasets x generated by different parameter vectors theta",
            size=26,
            weight=760,
        ),
        svg_text(
            MARGIN,
            72,
            "Dots are noisy observations; solid lines are the noiseless mean A exp(-k t).",
            size=16,
            color="#475066",
        ),
    ]

    for (scenario, points), (left, top) in zip(simulated, panel_positions, strict=True):
        amplitude, decay_rate, noise = scenario["theta"]
        parts.extend(
            make_panel(
                left=left,
                top=top,
                panel_width=panel_width,
                panel_height=panel_height,
                title=scenario["title"],
                theta_label=f"theta = (A={amplitude:g}, k={decay_rate:g}, sigma={noise:g})",
                times=times,
                points=points,
                amplitude=amplitude,
                decay_rate=decay_rate,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                color=scenario["color"],
            )
        )

    parts.append("</svg>")
    OUTFILE.write_text("\n".join(parts), encoding="utf-8")
    print(OUTFILE)


if __name__ == "__main__":
    main()
