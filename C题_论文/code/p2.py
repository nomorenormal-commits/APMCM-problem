"""Problem 2: acoustic fidelity, degradation models, effective communication range."""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, brentq
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import (DIST, CTYPES, CALLS, MEASURES, load_degradation, logistic_decay,
                    fit_decay, decay_range, r2, aicc, set_style, save, write_table, dump)

set_style()
rng = np.random.RandomState(20261126)
NBOOT = 1200

long = load_degradation()

# replicate arrays per call type
C = {}
for ct in CTYPES:
    C[ct] = {}
    for m in MEASURES:
        sub = long[(long.calltype == ct) & (long.measure == m)]
        mat = np.full((300, 7), np.nan)
        for j, d in enumerate(DIST):
            v = sub.loc[sub.distance == d, "corr"].values
            mat[:len(v), j] = v
        C[ct][m] = mat

MEAN = {(ct, m): np.nanmean(C[ct][m], axis=0) for ct in CTYPES for m in MEASURES}
SD = {(ct, m): np.nanstd(C[ct][m], axis=0, ddof=1) for ct in CTYPES for m in MEASURES}


def fisher_composite(vals, weights):
    """vals: (..., 3) array of correlations; returns weighted Fisher-z mean."""
    z = np.arctanh(np.clip(vals, 1e-4, 1 - 1e-6))
    return np.tanh(np.tensordot(z, weights, axes=([-1], [0])))


W = np.array([1 / 3, 1 / 3, 1 / 3])
MEAS_ORDER = ["spec", "amp", "env"]
COMP = {}
for ct in CTYPES:
    vals = np.column_stack([MEAN[(ct, m)] for m in MEAS_ORDER])
    COMP[ct] = fisher_composite(vals, W)


def fit_mean_curve(d, y, p0=None):
    if p0 is None:
        p0 = [max(y[0], .5), .05, 60, 2]
    popt, pcov = curve_fit(logistic_decay, d, y, p0=p0,
                           bounds=([0, 0, .5, .1], [1.05, .6, 5000, 12]), maxfev=100000)
    return popt


PARAM = {ct: fit_mean_curve(DIST, COMP[ct]) for ct in CTYPES}

# replicate-level composite values (used for model comparison at observation level)
CREP = {}
for ct in CTYPES:
    z = np.stack([np.arctanh(np.clip(C[ct][m], 1e-4, 1 - 1e-6)) for m in MEAS_ORDER])
    CREP[ct] = np.tanh(np.nanmean(z, axis=0))  # (300, 7)

# ---------------- model comparison ----------------
def solve_range(fn, p, T, dmax=2000.0):
    if not (fn(dmax, *p) < T < fn(4, *p)):
        return np.nan
    return brentq(lambda d: fn(d, *p) - T, 4.0, dmax)


def m_exp(d, f0, lam):
    return f0 * np.exp(-lam * d)


def m_power(d, f0, d0, k):
    return f0 * (1 + d / d0) ** (-k)


def m_weib(d, fmin, f0, lam, k):
    return fmin + (f0 - fmin) * np.exp(-(d / lam) ** k)


models = {
    "Exponential (2)": (m_exp, [0.8, 0.01], ([0, 0], [1.2, 0.2])),
    "Power law (3)": (m_power, [0.8, 20, 0.7], ([0, 0.1, 0.05], [1.2, 5000, 6])),
    "Hill/logistic (4)": (logistic_decay, [0.85, 0.05, 60, 2], ([0, 0, .5, .1], [1.05, .6, 5000, 12])),
    "Weibull (4)": (m_weib, [0.05, 0.85, 60, 1.2], ([0, 0, .5, .1], [1, 1.05, 5000, 12])),
}
comp_rows, rangemodel_rows = [], []
comp_res = {}
for name, (fn, p0, bnds) in models.items():
    row = [name]; rr = [name]
    r2s, aics = [], []
    for ct in CTYPES:
        x = np.tile(DIST, CREP[ct].shape[0])
        yv = CREP[ct].ravel()
        ok = np.isfinite(yv); x, yv = x[ok], yv[ok]
        try:
            p, _ = curve_fit(fn, x, yv, p0=p0, bounds=bnds, maxfev=200000)
            yy = fn(x, *p)
            row.append(f"{aicc(yv, yy, len(p)):.0f} / {r2(yv, yy):.3f}")
            aics.append(aicc(yv, yy, len(p))); r2s.append(r2(yv, yy))
            rr.append(f"{solve_range(fn, p, 0.70):.0f}")
        except Exception:
            row.append("--"); rr.append("--")
    comp_rows.append(row)
    rangemodel_rows.append(rr)
    comp_res[name] = dict(aicc=aics, r2=r2s)
write_table("tab_p2_models", ["Model (parameters)", "Flat", "Modulated", "Two-voice"], comp_rows,
            "Fit of four candidate degradation families to all 2,100 replicate fidelity values; each cell reports AICc / $R^2$.",
            "tab:p2models", "lccc")
write_table("tab_p2_modelfamily", ["Model (parameters)", "Flat", "Modulated", "Two-voice"], rangemodel_rows,
            "Reliable-recognition range (m, $T=0.70$) implied by each candidate degradation family.",
            "tab:p2mf", "lccc", place="[H]")

n = len(DIST)
summary_rows = []
for ct in CTYPES:
    f0, fmin, dh, k = PARAM[ct]
    yhat = logistic_decay(DIST, *PARAM[ct])
    summary_rows.append([CALLS[ct]["label"], f"{f0:.3f}", f"{fmin:.3f}", f"{dh:.1f}",
                         f"{k:.2f}", f"{r2(COMP[ct], yhat):.3f}",
                         f"{COMP[ct][0]:.3f}", f"{COMP[ct][-1]:.3f}"])
write_table("tab_p2_params", ["Call type", "$F_0$", "$F_{\\min}$", "$d_h$ (m)", "$\\kappa$", "$R^2$",
                              "$F(4\\,\\mathrm{m})$", "$F(256\\,\\mathrm{m})$"], summary_rows,
            "Estimated parameters of the composite fidelity curve $F(d)=F_{\\min}+(F_0-F_{\\min})/[1+(d/d_h)^{\\kappa}]$.",
            "tab:p2params", "lccccccc")

# composite table by distance
comp_tab = []
for j, d in enumerate(DIST):
    comp_tab.append([f"{int(d)}"] + [f"{COMP[ct][j]:.3f}" for ct in CTYPES])
write_table("tab_p2_composite", ["Distance (m)", "Flat", "Modulated", "Two-voice"], comp_tab,
            "Composite acoustic fidelity at each propagation distance (Fisher-$z$ average of the three correlations).",
            "tab:p2comp", "lccc", place="[H]")

# ---------------- bootstrap ranges ----------------
THRESH = [0.5, 0.6, 0.7]
BOOT = {ct: {t: [] for t in THRESH} for ct in CTYPES}
BOOTP = {ct: [] for ct in CTYPES}
for b in range(NBOOT):
    for ct in CTYPES:
        comp_b = np.empty(7)
        for j in range(7):
            vals = []
            for m in MEAS_ORDER:
                col = C[ct][m][:, j]
                col = col[~np.isnan(col)]
                vals.append(np.mean(rng.choice(col, size=len(col), replace=True)))
            comp_b[j] = fisher_composite(np.array(vals), W)
        try:
            p = fit_mean_curve(DIST, comp_b, p0=PARAM[ct])
            BOOTP[ct].append(p)
            for t in THRESH:
                BOOT[ct][t].append(decay_range(p, t))
        except Exception:
            continue

range_rows = []
range_res = {}
for ct in CTYPES:
    row = [CALLS[ct]["label"]]
    range_res[ct] = {}
    for t in THRESH:
        vals = np.array(BOOT[ct][t], dtype=float)
        vals = vals[np.isfinite(vals)]
        point = decay_range(PARAM[ct], t)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        row.append(f"{point:.0f} [{lo:.0f}, {hi:.0f}]")
        range_res[ct][t] = dict(point=point, lo=lo, hi=hi)
    range_rows.append(row)
write_table("tab_p2_ranges", ["Call type", "$T=0.50$ (potential)", "$T=0.60$ (recognition)",
                              "$T=0.70$ (reliable)"], range_rows,
            "Effective communication range (m) for three recognition thresholds; point estimate from the full data and 95\\% bootstrap interval in brackets.",
            "tab:p2ranges", "lccc")

# ---------------- sensitivity ----------------
sens_t = np.arange(0.40, 0.801, 0.05)
sens_rows = []
for t in sens_t:
    vals = []
    for ct in CTYPES:
        r = decay_range(PARAM[ct], t)
        vals.append(f"{r:.0f}" if np.isfinite(r) else "--")
    sens_rows.append([f"{t:.2f}"] + vals)
write_table("tab_p2_sens", ["Threshold $T$", "Flat", "Modulated", "Two-voice"], sens_rows,
            "Sensitivity of the effective range (m) to the recognition threshold.",
            "tab:p2sens", "lccc", place="[H]")

# leave-one-measure-out and equal-vs-arithmetic weights
robust_rows = []
for ct in CTYPES:
    base = decay_range(PARAM[ct], 0.7)
    for drop in MEAS_ORDER:
        w = np.array([0.5 if m != drop else 0.0 for m in MEAS_ORDER])
        w = w / w.sum()
        y = fisher_composite(np.column_stack([MEAN[(ct, m)] for m in MEAS_ORDER]), w)
        p = fit_mean_curve(DIST, y)
        robust_rows.append([CALLS[ct]["label"], f"drop {MEASURES[drop]['label'].lower()}",
                            f"{decay_range(p,0.7):.0f}"])
    # arithmetic mean instead of Fisher z
    y = np.column_stack([MEAN[(ct, m)] for m in MEAS_ORDER]).mean(axis=1)
    p = fit_mean_curve(DIST, y)
    robust_rows.append([CALLS[ct]["label"], "arithmetic mean", f"{decay_range(p,0.7):.0f}"])
write_table("tab_p2_robust", ["Call type", "Variant", "Range at $T=0.70$ (m)"], robust_rows,
            "Robustness of the reliable-recognition range to the construction of the composite index.",
            "tab:p2robust", "llc", place="[H]")

# ---------------- figures ----------------
# individual measures
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), sharey=True)
for ax, ct in zip(axes, CTYPES):
    for m in MEAS_ORDER:
        info = MEASURES[m]
        ax.errorbar(DIST, MEAN[(ct, m)], yerr=SD[(ct, m)], fmt=info["marker"],
                    color=info["color"], ms=3.5, lw=0.8, capsize=2, alpha=0.85,
                    label=info["label"])
        yy = np.linspace(4, 256, 200)
        p = fit_mean_curve(DIST, MEAN[(ct, m)])
        ax.plot(yy, logistic_decay(yy, *p), color=info["color"], lw=1.1, alpha=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIST); ax.set_xticklabels([str(int(d)) for d in DIST], fontsize=7)
    ax.set_xlabel("Distance (m)")
    ax.set_title(CALLS[ct]["label"], color=CALLS[ct]["color"])
    for t in (0.5, 0.6, 0.7):
        ax.axhline(t, color="0.55", lw=0.55, ls="--", zorder=0)
        if ct == CTYPES[0]:
            ax.text(4.4, t + 0.012, f"T={t:.1f}", fontsize=5.2, color="0.35")
    ax.set_ylim(0, 1)
axes[0].set_ylabel("Correlation with 1 m reference")
axes[2].legend(loc="lower left", fontsize=7)
fig.tight_layout()
save(fig, "fig_p2_measures")

# composite curves
fig, ax = plt.subplots(figsize=(4.5, 3.1))
xx = np.logspace(np.log10(4), np.log10(256), 300)
for ct in CTYPES:
    p = PARAM[ct]
    ax.plot(xx, logistic_decay(xx, *p), color=CALLS[ct]["color"], lw=1.5,
            label=CALLS[ct]["label"])
    ax.plot(DIST, COMP[ct], CALLS[ct]["marker"], color=CALLS[ct]["color"], ms=3.5)
    # bootstrap band
    bp = np.array(BOOTP[ct])
    band = np.array([logistic_decay(xx, *p) for p in bp[::20]])
    lo, hi = np.percentile(band, [5, 95], axis=0)
    ax.fill_between(xx, lo, hi, color=CALLS[ct]["color"], alpha=0.12, lw=0)
tlabels = {0.5: "potential", 0.6: "recognition", 0.7: "reliable"}
for t, ls in zip(THRESH, [":", "--", "-."]):
    ax.axhline(t, color="grey", lw=0.7, ls=ls)
    ax.text(4.3, t + 0.012, f"$T={t}$ {tlabels[t]}", ha="left", fontsize=6.5, color="grey")
ax.set_xscale("log", base=2)
ax.set_xticks(DIST); ax.set_xticklabels([str(int(d)) for d in DIST])
ax.set_xlabel("Distance (m)"); ax.set_ylabel("Composite fidelity $F(d)$")
ax.set_ylim(0, 1); ax.legend(loc="lower left", fontsize=7)
fig.tight_layout()
save(fig, "fig_p2_composite")

# ranges with uncertainty
fig, ax = plt.subplots(figsize=(3.4, 2.7))
w = 0.26
for i, t in enumerate(THRESH):
    xs = np.arange(3) + (i - 1) * w
    vals = [range_res[ct][t]["point"] for ct in CTYPES]
    lo = [range_res[ct][t]["point"] - range_res[ct][t]["lo"] for ct in CTYPES]
    hi = [range_res[ct][t]["hi"] - range_res[ct][t]["point"] for ct in CTYPES]
    ax.bar(xs, vals, width=w, color=["#9ecae1", "#6baed6", "#2171b5"][i],
           yerr=[lo, hi], capsize=2, edgecolor="none",
           label=f"$T={t}$")
ax.set_xticks(range(3)); ax.set_xticklabels(["Flat", "Mod.", "2-voice"])
ax.set_ylabel("Effective range (m)"); ax.legend(fontsize=7)
fig.tight_layout()
save(fig, "fig_p2_ranges")

# sensitivity heat map
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2), sharey=True)
for ax, ct in zip(axes, CTYPES):
    z = np.array([[decay_range(PARAM[ct], t) for t in np.arange(0.4, 0.81, 0.02)]])
    im = ax.imshow(z, aspect="auto", cmap="viridis", origin="lower",
                   extent=[0.4, 0.8, 0, 1], vmin=0, vmax=180)
    ax.set_title(CALLS[ct]["label"], color=CALLS[ct]["color"])
    ax.set_xlabel("Threshold $T$"); ax.set_yticks([])
axes[0].set_ylabel("")
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
cb.set_label("Range (m)")
save(fig, "fig_p2_sensitivity")

dump("p2", dict(param={ct: PARAM[ct].tolist() for ct in CTYPES},
                comp={ct: COMP[ct].tolist() for ct in CTYPES},
                ranges={ct: {str(k): v for k, v in range_res[ct].items()} for ct in CTYPES},
                model_compare=comp_res))
print(open(__import__("common").RES / "p2.json").read())
