"""Create Figure S1 as a Robinson-map, pseudo-3D mechanism schematic."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

INK = "#203342"
MUTED = "#607381"
OCEAN = "#dcecf1"
LAND = "#eee9dc"
BLUE_DARK = "#315f7d"
BLUE = "#5d91ad"
BLUE_LIGHT = "#b9dbe5"
GOLD = "#d7a947"
GREEN = "#5a9474"
PURPLE = "#84638b"
RED = "#b85d49"


def bezier_loop(y_offset=0.0):
    """Raised overturning-style loop in axes coordinates."""
    vertices = [
        (0.22, 0.53 + y_offset),
        (0.39, 0.49 + y_offset),
        (0.62, 0.49 + y_offset),
        (0.78, 0.55 + y_offset),
        (0.80, 0.60 + y_offset),
        (0.80, 0.75 + y_offset),
        (0.77, 0.80 + y_offset),
        (0.66, 0.85 + y_offset),
        (0.34, 0.85 + y_offset),
        (0.22, 0.80 + y_offset),
        (0.20, 0.74 + y_offset),
        (0.20, 0.59 + y_offset),
        (0.22, 0.53 + y_offset),
        (0.22, 0.53 + y_offset),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(vertices, codes)


def add_ribbon(ax):
    """Draw an elevated, pseudo-3D circulation loop."""
    shadow = PathPatch(
        bezier_loop(-0.025),
        transform=ax.transAxes,
        fill=False,
        linewidth=28,
        edgecolor="#1d3546",
        alpha=0.18,
        capstyle="round",
        joinstyle="round",
        zorder=12,
    )
    side = PathPatch(
        bezier_loop(-0.010),
        transform=ax.transAxes,
        fill=False,
        linewidth=23,
        edgecolor=BLUE_DARK,
        alpha=0.72,
        capstyle="round",
        joinstyle="round",
        zorder=14,
    )
    top = PathPatch(
        bezier_loop(0.0),
        transform=ax.transAxes,
        fill=False,
        linewidth=18,
        edgecolor=BLUE,
        alpha=0.92,
        capstyle="round",
        joinstyle="round",
        zorder=15,
    )
    highlight = PathPatch(
        bezier_loop(0.008),
        transform=ax.transAxes,
        fill=False,
        linewidth=4,
        edgecolor=BLUE_LIGHT,
        alpha=0.92,
        capstyle="round",
        joinstyle="round",
        zorder=16,
    )
    for patch in (shadow, side, top, highlight):
        ax.add_patch(patch)

    # Arrowheads establish the closed-loop reading without implying a reconstructed path.
    for start, end in [
        ((0.44, 0.505), (0.57, 0.515)),
        ((0.79, 0.63), (0.79, 0.73)),
        ((0.59, 0.835), (0.46, 0.84)),
        ((0.205, 0.72), (0.21, 0.61)),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=2.2,
                color=BLUE_LIGHT,
                zorder=18,
            )
        )

    # Short supports and a soft floor shadow create height.
    for x0, y0 in [(0.28, 0.52), (0.43, 0.50), (0.60, 0.51), (0.74, 0.54)]:
        ax.plot(
            [x0, x0],
            [y0 - 0.07, y0 - 0.015],
            transform=ax.transAxes,
            color=BLUE_DARK,
            alpha=0.28,
            linewidth=1.0,
            linestyle=(0, (2, 3)),
            zorder=11,
        )

    ax.text(
        0.47,
        0.71,
        "IPO-associated slow climatic background",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=INK,
        zorder=19,
    )
    ax.text(
        0.47,
        0.675,
        "conceptual circulation reorganization",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color=BLUE_DARK,
        zorder=19,
    )


def raised_pill(ax, center, size, text, color, fontsize=7.5):
    """Floating process label with a small offset shadow."""
    x, y = center
    width, height = size
    shadow = FancyBboxPatch(
        (x - width / 2 + 0.008, y - height / 2 - 0.010),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor="#183242",
        edgecolor="none",
        alpha=0.16,
        zorder=21,
    )
    pill = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor=color,
        linewidth=1.15,
        alpha=0.96,
        zorder=22,
    )
    ax.add_patch(shadow)
    ax.add_patch(pill)
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=INK,
        zorder=23,
    )


def dashed_link(ax, start, end, color=MUTED, curve=0.0, head=8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=head,
            linewidth=1.1,
            linestyle=(0, (4, 3)),
            color=color,
            connectionstyle=f"arc3,rad={curve}",
            zorder=24,
        )
    )


def starburst(center, outer=0.035, inner=0.017, spikes=12):
    angles = np.linspace(0, 2 * np.pi, spikes * 2, endpoint=False) + np.pi / 2
    radii = np.where(np.arange(spikes * 2) % 2 == 0, outer, inner)
    return np.column_stack([center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)])


def main():
    # Figure contract
    # Conclusion: IPO-associated background variability is consistent with a
    # susceptibility window whose regional expression may involve hydroclimatic,
    # productivity, circulation, and event-scale contexts.
    # Archetype: single-hero pseudo-3D mechanism map.
    # Risk boundary: the elevated loop is conceptual, not a reconstructed 3D flow.
    fig = plt.figure(figsize=(297 / 25.4, 210 / 25.4), facecolor="white")
    ax = fig.add_axes([0.04, 0.14, 0.92, 0.74], projection=ccrs.Robinson(central_longitude=180))
    ax.set_extent([100, 300, -50, 35], crs=ccrs.PlateCarree())
    ax.set_facecolor(OCEAN)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND, edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=INK, linewidth=0.5, zorder=3)
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.35,
        color="#76909e",
        alpha=0.40,
        linestyle=":",
        xlocs=np.arange(120, 301, 30),
        ylocs=np.arange(-45, 31, 15),
    )

    fig.suptitle(
        "Hypothesized IPO-Conditioned Susceptibility to GBR COTS Outbreaks",
        x=0.50,
        y=0.965,
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.50,
        0.965,
        "Pacific basin, Robinson projection (latitude-cropped)",
        transform=ax.transAxes,
        fontsize=7.5,
        color=MUTED,
        ha="center",
        va="top",
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", boxstyle="round,pad=0.22"),
        zorder=30,
    )

    add_ribbon(ax)

    # The elevated basin-scale loop conditions, rather than directly triggers,
    # the lower susceptibility lens.
    dashed_link(ax, (0.43, 0.49), (0.49, 0.355), color=BLUE_DARK, curve=-0.06, head=9)
    dashed_link(ax, (0.57, 0.50), (0.53, 0.355), color=BLUE_DARK, curve=0.06, head=9)
    ax.text(
        0.51,
        0.422,
        "modulates background susceptibility",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.0,
        color=BLUE_DARK,
        bbox=dict(facecolor=OCEAN, alpha=0.92, edgecolor="none", boxstyle="round,pad=0.18"),
        zorder=31,
    )

    # GBR is the visual landing point.
    gbr_lon, gbr_lat = 147.5, -18.3
    ax.scatter(
        [gbr_lon],
        [gbr_lat],
        transform=ccrs.PlateCarree(),
        s=320,
        color=GOLD,
        alpha=0.18,
        edgecolor="none",
        zorder=27,
    )
    ax.scatter(
        [gbr_lon],
        [gbr_lat],
        transform=ccrs.PlateCarree(),
        s=55,
        color=GOLD,
        edgecolor="white",
        linewidth=1.1,
        zorder=29,
    )
    ax.annotate(
        "Great Barrier Reef",
        xy=(gbr_lon, gbr_lat),
        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
        xytext=(0.19, 0.31),
        textcoords=ax.transAxes,
        ha="right",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=INK,
        arrowprops=dict(arrowstyle="-", color=GOLD, linewidth=1.2),
        zorder=31,
    )

    # Candidate contexts float around, rather than forming a deterministic chain.
    raised_pill(ax, (0.25, 0.105), (0.17, 0.060), "Rainfall and runoff", "#397fa6")
    raised_pill(ax, (0.46, 0.085), (0.18, 0.060), "Shelf productivity", GREEN)
    raised_pill(ax, (0.69, 0.105), (0.22, 0.060), "Signed meridional-flow contrast", "#5f70a5", fontsize=7.1)

    # Central susceptibility lens.
    ax.add_patch(
        Circle(
            (0.52, 0.30),
            0.105,
            transform=ax.transAxes,
            facecolor=GOLD,
            edgecolor="none",
            alpha=0.12,
            zorder=20,
        )
    )
    raised_pill(
        ax,
        (0.52, 0.30),
        (0.28, 0.085),
        "IPO-conditioned susceptibility window",
        GOLD,
        fontsize=8.5,
    )

    for start in [(0.25, 0.14), (0.46, 0.12), (0.69, 0.14)]:
        dashed_link(ax, start, (0.52, 0.255), curve=0.10 if start[0] < 0.52 else -0.10)
    ax.text(
        0.52,
        0.195,
        "candidate pathways",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.6,
        color=MUTED,
        bbox=dict(facecolor=OCEAN, alpha=0.92, edgecolor="none", boxstyle="round,pad=0.16"),
        zorder=31,
    )

    # Release/amplification is distinct from low-frequency conditioning.
    raised_pill(ax, (0.82, 0.30), (0.16, 0.068), "ENSO-scale /\nlocal events", PURPLE, fontsize=7.0)
    dashed_link(ax, (0.74, 0.30), (0.66, 0.30), color=PURPLE, curve=0.06)
    ax.text(
        0.70,
        0.358,
        "may release\nor amplify",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=PURPLE,
        zorder=31,
    )

    # Outbreak icon and ecological phase ordering are separate from flow.
    star_xy = (0.28, 0.27)
    ax.add_patch(
        Polygon(
            starburst(star_xy),
            closed=True,
            transform=ax.transAxes,
            facecolor=RED,
            edgecolor="white",
            linewidth=0.9,
            zorder=30,
        )
    )
    ax.text(
        0.28,
        0.215,
        "Later adult COTS\noutbreak expression",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=INK,
        zorder=31,
    )
    dashed_link(ax, (0.38, 0.30), (0.315, 0.275), color=RED, curve=0.08, head=10)

    # A compact evidence-boundary note replaces the prior flowchart disclaimer.
    fig.text(
        0.50,
        0.055,
        "Dashed links represent hypothesized connections among climate, local conditions, and outbreaks.",
        ha="center",
        va="center",
        fontsize=8,
        color=MUTED,
    )

    from publication_style import save_publication
    stem = OUTPUT_DIR / "FigureS1_Conceptual_Susceptibility"
    save_publication(fig, stem, orientation="landscape")
    fig.savefig(stem.with_suffix(".tiff"), dpi=300, bbox_inches=None, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
