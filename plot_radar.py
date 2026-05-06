import math
from pathlib import Path


METHODS = ["VLP", "uniform", "confidence-r"]
LEGEND_METHODS = ["uniform", "confidence-r", "VLP"]
STEPS = [32, 64]

# Values from Table 1 (RADD perplexity) and Table 2 (Dream-7B scores).
RAW = {
    64: {
        "VLP": {
            "PPL": 138.0,
            "HumanEval": 27.4,
            "MBPP": 31.6,
            "GSM8K": 38.3,
            "MATH500": 17.2,
        },
        "uniform": {
            "PPL": 154.5,
            "HumanEval": 19.5,
            "MBPP": 27.6,
            "GSM8K": 34.9,
            "MATH500": 13.8,
        },
        "confidence-r": {
            "PPL": 140.7,
            "HumanEval": 26.2,
            "MBPP": 27.4,
            "GSM8K": 39.2,
            "MATH500": 16.6,
        },
    },
    32: {
        "VLP": {
            "PPL": 174.8,
            "HumanEval": 22.6,
            "MBPP": 26.2,
            "GSM8K": 36.6,
            "MATH500": 17.0,
        },
        "uniform": {
            "PPL": 199.7,
            "HumanEval": 17.6,
            "MBPP": 25.6,
            "GSM8K": 30.0,
            "MATH500": 12.8,
        },
        "confidence-r": {
            "PPL": 181.6,
            "HumanEval": 18.3,
            "MBPP": 24.4,
            "GSM8K": 36.5,
            "MATH500": 16.8,
        },
    },
}

AXES = [
    ("PPL", "GenPPL", "down"),
    ("HumanEval", "HumanEval", "up"),
    ("MBPP", "MBPP", "up"),
    ("GSM8K", "GSM8K", "up"),
    ("MATH500", "MATH500", "up"),
]

COLORS = {
    "VLP": "#2F80ED",
    "uniform": "#555555",
    "confidence-g": "#E07A5F",
    "confidence-r": "#59A14F",
}


def normalize(value, worst, best, direction):
    if math.isclose(worst, best):
        return 1.0
    return 0.25 + 0.75 * (value - worst) / (best - worst)


def axis_scales():
    scales = {}
    for key, _, direction in AXES:
        values = [RAW[step][method][key] for step in STEPS for method in METHODS]
        if direction == "down":
            best = min(values)
            worst = max(values)
        else:
            best = max(values)
            worst = min(values)
        scales[key] = (worst, best, direction)
    return scales


def normalized_values(step, method, scales):
    return [
        normalize(RAW[step][method][key], *scales[key])
        for key, _, _ in AXES
    ]


def save_matplotlib(out_dir):
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    import numpy as np
    from matplotlib.lines import Line2D

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    scales = axis_scales()
    angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(6.4, 5.1), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="#666666")
    ax.grid(color="#CFCFCF", linewidth=0.8)
    ax.spines["polar"].set_color("#B5B5B5")

    for method in LEGEND_METHODS:
        vals = normalized_values(64, method, scales)
        closed_values = np.concatenate([vals, vals[:1]])
        ax.plot(
            closed_angles,
            closed_values,
            color=COLORS[method],
            linewidth=2.2,
            linestyle="-",
            label=f"{method} (64)",
        )
        ax.fill(closed_angles, closed_values, color=COLORS[method], alpha=0.10)

        vals = normalized_values(32, method, scales)
        closed_values = np.concatenate([vals, vals[:1]])
        ax.plot(
            closed_angles,
            closed_values,
            color=COLORS[method],
            linewidth=2.2,
            linestyle=(0, (2.5, 2.0)),
            label=f"{method} (32)",
        )
        ax.fill(closed_angles, closed_values, color=COLORS[method], alpha=0.2)

    for angle, (key, label, _) in zip(angles, AXES):
        label_radius = 1.18 if key in {"HumanEval", "MATH500"} else 1.09
        if key == "HumanEval":
            label_radius = 1.24
        dx = -5 if key == "MATH500" else 0
        dy = -5
        text_transform = ax.transData + mtransforms.ScaledTranslation(
            dx / 72,
            dy / 72,
            fig.dpi_scale_trans,
        )
        ax.text(
            angle,
            label_radius,
            label,
            transform=text_transform,
            ha="center",
            va="center",
            fontsize=10,
        )

    # Matplotlib fills legend entries down columns. This interleaving produces
    # visible rows of: "32 steps: uniform confidence-r VLP" and then 64 steps.
    handles = [
        Line2D([], [], color="none", label="32 steps:"),
        Line2D([], [], color="none", label="64 steps:"),
    ]
    for method in LEGEND_METHODS:
        handles.extend(
            [
                Line2D(
                    [],
                    [],
                    color=COLORS[method],
                    linewidth=2.2,
                    linestyle=(0, (2.5, 2.0)),
                    label=method,
                ),
                Line2D(
                    [],
                    [],
                    color=COLORS[method],
                    linewidth=2.2,
                    linestyle="-",
                    label=method,
                ),
            ]
        )
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.2,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_dir / "radar_steps32_64.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "radar_steps32_64.pdf", bbox_inches="tight")


def svg_text(x, y, text, size=13, anchor="middle", weight="normal", color="#222222"):
    lines = text.split("\n")
    if len(lines) == 1:
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'font-family="Helvetica, Arial, sans-serif" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}">{lines[0]}</text>'
        )
    tspans = [
        (
            f'<tspan x="{x:.1f}" dy="{0 if i == 0 else size * 1.15:.1f}">'
            f"{line}</tspan>"
        )
        for i, line in enumerate(lines)
    ]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Helvetica, Arial, sans-serif" font-weight="{weight}" '
        f'fill="{color}" text-anchor="{anchor}">'
        + "".join(tspans)
        + "</text>"
    )


def points_to_svg(points):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def save_svg(out_dir):
    scales = axis_scales()
    width, height = 900, 650
    cx, cy = 450, 285
    radius = 230
    angles = [(-math.pi / 2) + 2 * math.pi * i / len(AXES) for i in range(len(AXES))]

    def point(angle, value):
        return cx + radius * value * math.cos(angle), cy + radius * value * math.sin(angle)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for frac in [0.25, 0.5, 0.75, 1.0]:
        ring = [point(angle, frac) for angle in angles]
        elements.append(
            f'<polygon points="{points_to_svg(ring)}" fill="none" '
            f'stroke="#CFCFCF" stroke-width="1"/>'
        )
        elements.append(svg_text(cx + 6, cy - radius * frac + 4, f"{int(frac * 100)}%", size=10, anchor="start", color="#666666"))

    for (key, label, direction), angle in zip(AXES, angles):
        x2, y2 = point(angle, 1.0)
        elements.append(
            f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#CFCFCF" stroke-width="1"/>'
        )
        offset = 88 if key in {"HumanEval", "MATH500"} else 64
        lx = cx + (radius + offset) * math.cos(angle)
        ly = cy + (radius + offset) * math.sin(angle)
        if key == "HumanEval":
            lx += 24
        anchor = "middle"
        if math.cos(angle) > 0.4:
            anchor = "start"
        elif math.cos(angle) < -0.4:
            anchor = "end"
        elements.append(svg_text(lx, ly, label, size=13, anchor=anchor))

    for method in METHODS:
        vals = normalized_values(64, method, scales)
        pts = [point(angle, val) for angle, val in zip(angles, vals)]
        elements.append(
            f'<polygon points="{points_to_svg(pts)}" fill="{COLORS[method]}" '
            f'fill-opacity="0.12" stroke="{COLORS[method]}" stroke-width="3"/>'
        )
        for x, y in pts:
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{COLORS[method]}"/>')

        vals = normalized_values(32, method, scales)
        pts = [point(angle, val) for angle, val in zip(angles, vals)]
        elements.append(
            f'<polygon points="{points_to_svg(pts)}" fill="{COLORS[method]}" '
            f'fill-opacity="0.06" stroke="none"/>'
        )
        elements.append(
            f'<polygon points="{points_to_svg(pts)}" fill="none" '
            f'stroke="{COLORS[method]}" stroke-width="3" stroke-dasharray="4 3" '
            f'stroke-linecap="round"/>'
        )
        for x, y in pts:
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="white" '
                f'stroke="{COLORS[method]}" stroke-width="2"/>'
            )

    legend_x = 170
    method_x = 290
    for row, (step_text, y, dash) in enumerate(
        [("32 steps:", 575, True), ("64 steps:", 615, False)]
    ):
        elements.append(svg_text(legend_x, y + 5, step_text, size=13, anchor="start"))
        for i, method in enumerate(LEGEND_METHODS):
            x = method_x + i * 155
            dash_attr = ' stroke-dasharray="4 3" stroke-linecap="round"' if dash else ""
            elements.append(
                f'<line x1="{x}" y1="{y}" x2="{x + 30}" y2="{y}" '
                f'stroke="{COLORS[method]}" stroke-width="4"{dash_attr}/>'
            )
            elements.append(svg_text(x + 40, y + 5, method, size=13, anchor="start"))

    elements.append("</svg>")
    (out_dir / "radar_steps32_64.svg").write_text("\n".join(elements), encoding="utf-8")


def main():
    out_dir = Path("figs")
    out_dir.mkdir(exist_ok=True)
    try:
        save_matplotlib(out_dir)
        print("Saved figs/radar_steps32_64.png and figs/radar_steps32_64.pdf")
    except ModuleNotFoundError:
        save_svg(out_dir)
        print("Matplotlib not found; saved figs/radar_steps32_64.svg")


if __name__ == "__main__":
    main()
