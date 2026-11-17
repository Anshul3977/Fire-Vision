import matplotlib

matplotlib.use("Agg")  # Set non-GUI backend before any other Matplotlib imports
import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np


def ordered_unique(points):
    """Drop duplicate path points without scrambling ignition vs. the front.

    list(set(points)) used to do that, so the star markers could swap.
    """
    seen = set()
    ordered = []
    for point in points:
        key = (int(point[0]), int(point[1]))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def graph(time_seconds, areas, smoke_dir, points, dpi=80):
    fig = None
    try:
        fig = plt.figure(figsize=(11, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_seconds, areas, label="Burned Area", color="firebrick")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Cumulative Area (pixel units)")
        ax1.set_title("Cumulative Burned Area Over Time")
        ax1.grid(True)
        ax1.legend()

        area_growth = [areas[i + 2] - areas[i + 1] for i in range(len(areas) - 2)]
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(time_seconds[2:], area_growth, label="Area Change", color="orangered")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Change in Area (pixel units)")
        ax2.set_title("Rate of Area Change Over Time")
        ax2.grid(True)
        ax2.legend()

        ax3 = fig.add_subplot(gs[0, 1], polar=True)
        finite_smoke = [float(value) for value in smoke_dir if np.isfinite(value)]
        if finite_smoke:
            ax3.hist(np.radians(finite_smoke), bins=30, color="steelblue", alpha=0.7)
        ax3.set_title("Smoke Dispersion Direction")
        ax3.set_theta_direction(-1)
        ax3.set_rticks([])

        ax4 = fig.add_subplot(gs[1, 1])
        ordered = ordered_unique(points)
        if ordered:
            x, y = zip(*ordered)
            ax4.plot(
                x, y, marker="o", color="darkorange", markersize=5, linestyle="-", label="Fire Path"
            )
            ax4.invert_yaxis()
            ax4.plot(
                x[0], y[0], marker="*", color="limegreen", markersize=12, label="Ignition Point"
            )
            ax4.plot(
                x[-1], y[-1], marker="*", color="red", markersize=12, label="Current Fire Front"
            )
            ax4.legend()

        ax4.set_xlabel("X Coordinate")
        ax4.set_ylabel("Y Coordinate")
        ax4.set_title("Fire Spread Path")
        ax4.grid(True)

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:
        print(f"Error in graph generation: {exc}")
        return None
    finally:
        if fig is not None:
            plt.close(fig)
