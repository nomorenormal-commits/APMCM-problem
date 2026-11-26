"""Threshold-conditional probability of the effective radius, from the saved Bayesian chain."""
import json
import numpy as np
import matplotlib.pyplot as plt

from common import CTYPES, CALLS, RES, set_style, save

set_style()
chain = np.load(RES / "p5_chain.npz")["chain"]
draws = chain[::9]                      # ~1,000 posterior draws
grid = np.arange(4, 301, 1.0)
thr = np.linspace(0.45, 0.75, 13)


def sig(x):
    return 1 / (1 + np.exp(-x))


rad = {}
surv = {}
for ct in CTYPES:
    i = CTYPES.index(ct)
    f0 = sig(draws[:, i])
    fmin = f0 * sig(draws[:, 3 + i])
    dh = np.exp(draws[:, 6 + i])
    kk = np.exp(draws[:, 9 + i])
    F = fmin[:, None] + (f0 - fmin)[:, None] / (1 + (grid[None, :] / dh[:, None]) ** kk[:, None])
    R = np.zeros((len(draws), len(thr)))
    for j, t in enumerate(thr):
        ok = F >= t
        for r in range(len(draws)):
            R[r, j] = grid[ok[r]].max() if ok[r].any() else 0.0
    rad[ct] = R
    surv[ct] = np.array([(R[:, j][:, None] >= grid[None, :]).mean(axis=0)
                         for j in range(len(thr))])

sel = {}
for ct in CTYPES:
    sel[ct] = {}
    for t in [0.50, 0.60, 0.70]:
        j = int(np.argmin(abs(thr - t)))
        sel[ct][f"T={t:.2f}"] = {str(d): float((rad[ct][:, j] >= d).mean()) for d in [100, 120, 150]}

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), constrained_layout=True)
for ax, ct in zip(axes, CTYPES):
    im = ax.pcolormesh(grid, thr, surv[ct], shading="auto", cmap="viridis", vmin=0, vmax=1)
    cs = ax.contour(grid, thr, surv[ct], levels=[0.5], colors="black", linewidths=0.8)
    ax.clabel(cs, fmt="0.5", fontsize=6)
    ax.set_xlabel("Distance $d$ (m)")
    ax.set_title(CALLS[ct]["label"], color=CALLS[ct]["color"])
    ax.set_xlim(4, 300)
    ax.set_ylim(0.45, 0.75)
axes[0].set_ylabel("Threshold $T$")
cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
cbar.set_label("$P(R_k \\geq d \\mid T)$", fontsize=7)
save(fig, "fig_p5_threshold")

(RES / "p5_threshold.json").write_text(json.dumps(sel, indent=1))
print(json.dumps(sel, indent=1))
