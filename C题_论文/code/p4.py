"""Problem 4: acoustic communication effectiveness, radii and noise scenarios."""
import json
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from common import (DATA, CALLS, CTYPES, DIST, logistic_decay, decay_range,
                    set_style, save, write_table, dump)

set_style()
rng = np.random.RandomState(2026)

P2 = json.loads((__import__("common").RES / "p2.json").read_text())
PARAM = {ct: np.array(P2["param"][ct]) for ct in CTYPES}

d = pd.read_csv(DATA / "Data02_AfricanPenguins_playbacks_data.csv")
d["CallType"] = pd.Categorical(d["CallType"], categories=["control", "flat", "mod", "tv"], ordered=True)
d["Context"] = pd.Categorical(d["Context"], categories=["commuting", "searching", "after"], ordered=True)
d["log2D"] = np.log2(d["Distance"])
d["log2N"] = np.log2(d["Npenguins"])

RESP = smf.logit("ResponseYesNo ~ C(CallType, Treatment('control')) + C(Context, Treatment('commuting')) + log2D + log2N",
                 data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["expID"]}, maxiter=500)
DI = RESP.model.data.orig_exog.design_info


def pred_p(ct, dist, n=8, context="searching"):
    if np.ndim(dist) == 0:
        nd = pd.DataFrame({"CallType": [ct], "Context": [context], "Distance": [dist],
                           "log2D": [np.log2(dist)], "log2N": [np.log2(n)], "Npenguins": [n]})
    else:
        nd = pd.DataFrame({"CallType": ct, "Context": context, "Distance": dist,
                           "log2D": np.log2(dist), "log2N": np.log2(n), "Npenguins": n})
    X = patsy.build_design_matrices([DI], nd, return_type="dataframe")[0]
    out = np.asarray(RESP.predict(nd)).reshape(-1)
    return out if out.size > 1 else float(out[0])


def fidelity(ct, dist, delta=0.0):
    return (1 - delta) * logistic_decay(dist, *PARAM[ct])


def index(ct, dist, delta=0.0, n=8, context="searching"):
    return np.sqrt(fidelity(ct, dist, delta) * pred_p(ct, dist, n, context))


def cross(fn, target, lo=4.0, hi=400.0):
    if fn(lo) < target:
        return np.nan
    if fn(hi) >= target:
        return hi
    return brentq(lambda x: fn(x) - target, lo, hi)


ISTAR = 0.5
N_TYP = 8
PHI = 0.5


def radius(ct, delta=0.0, tcap=0.5, istar=ISTAR):
    r = cross(lambda x: index(ct, x, delta), istar)
    if tcap is not None:
        p = np.array(PARAM[ct], dtype=float)
        p[0] *= (1 - delta)
        p[1] *= (1 - delta)
        cap = decay_range(p, tcap)
        r = 0.0 if not np.isfinite(cap) else min(r, cap)
    return r


# ---------------- component table ----------------
comp_rows = []
for ct in CTYPES:
    for dist in [50, 100, 150]:
        F = fidelity(ct, dist)
        P = float(pred_p(ct, dist))
        I = np.sqrt(F * P)
        E = N_TYP * PHI * P
        comp_rows.append([CALLS[ct]["label"], str(dist), f"{F:.3f}", f"{P:.3f}",
                          f"{I:.3f}", f"{E:.2f}"])
write_table("tab_p4_components", ["Call type", "Distance (m)", "Fidelity $F$", "P(response)",
                                  "Index $I$", "Expected responders"], comp_rows,
            "Components of the communication-effectiveness index for a typical group ($N=8$, $\\phi=0.5$, searching context).",
            "tab:p4comp", "llcccc")

# ---------------- radii ----------------
rad_rows = []
rad_res = {}
for ct in CTYPES:
    base = radius(ct)
    rad_res[ct] = dict(base=base)
    rad_rows.append([CALLS[ct]["label"], f"{base:.0f}",
                     f"{radius(ct, tcap=0.60):.0f}", f"{radius(ct, tcap=0.70):.0f}",
                     f"{radius(ct, delta=0.10):.0f}", f"{radius(ct, delta=0.20):.0f}",
                     f"{radius(ct, delta=0.10, tcap=0.60):.0f}"])
write_table("tab_p4_radius", ["Call type", "Baseline", "$T=0.60$", "$T=0.70$",
                              "$-10\\%$", "$-20\\%$", "$-10\\%$, $T=0.60$"],
            rad_rows,
            "Effective communication radius (m): the largest distance satisfying both $I\\ge0.50$ and $F\\ge T$, under the baseline and five degradation scenarios.",
            "tab:p4radius", "lcccccc")

# ---------------- weight and binding-constraint robustness ----------------
def index_alpha(ct, dist, alpha, delta=0.0):
    F = fidelity(ct, dist, delta)
    P = float(pred_p(ct, dist))
    if alpha == 0.0:
        return P
    if alpha == 1.0:
        return F
    return F ** alpha * P ** (1 - alpha)


w_rows = []
w_res = {}
for a in [0.0, 0.3, 0.5, 0.7, 1.0]:
    row = [f"{a:.1f}"]
    w_res[str(a)] = {}
    for ct in CTYPES:
        dI = cross(lambda x: index_alpha(ct, x, a), ISTAR)
        w_res[str(a)][ct] = float(dI)
        row.append(f"{dI:.0f}")
    w_rows.append(row)
write_table("tab_p4_weights", ["Weight $\\alpha$", "$d_I$ flat", "$d_I$ modulated",
                               "$d_I$ two-voice"], w_rows,
            "Distance (m) at which $I_\\alpha=F^{\\alpha}P^{1-\\alpha}=0.5$ for a grid of weights; the reported radius is $\\min(d_F,d_I)$ with $d_F$ the $F\\ge0.50$ radius.",
            "tab:p4weights", "lccc")

# ---------------- sensitivity to phi and group size ----------------
phi_rows = []
for ct in CTYPES:
    row = [CALLS[ct]["label"]]
    for phi in [0.25, 0.50, 0.75]:
        # radius is independent of phi; report expected responders at 100 m instead
        P = float(pred_p(ct, 100))
        row.append(f"{N_TYP*phi*P:.2f}")
    phi_rows.append(row)
write_table("tab_p4_phi", ["Call type", "$\\phi=0.25$", "$\\phi=0.50$", "$\\phi=0.75$"], phi_rows,
            "Expected number of responding individuals at 100 m ($N=8$) under three assumptions about the proportion $\\phi$ of responders.",
            "tab:p4phi", "lccc", place="[H]")

group_rows = []
for ct in CTYPES:
    row = [CALLS[ct]["label"]]
    for n in [3, 8, 20]:
        P = float(pred_p(ct, 100, n=n))
        row.append(f"{n*PHI*P:.2f}")
    group_rows.append(row)
write_table("tab_p4_group", ["Call type", "$N=3$", "$N=8$", "$N=20$"], group_rows,
            "Expected number of responding individuals at 100 m ($\\phi=0.5$) for three group sizes.",
            "tab:p4group", "lccc", place="[H]")

# ---------------- noise / threshold grid ----------------
grid_rows = []
for delta in [0.0, 0.05, 0.10, 0.15, 0.20]:
    grid_rows.append([f"{int(delta*100)}\\%",
                      *[f"{radius(ct, delta=delta, tcap=0.60):.0f}" for ct in CTYPES]])
write_table("tab_p4_noise", ["Correlation loss", "Flat", "Modulated", "Two-voice"], grid_rows,
            "Effective radius (m) when the recognition threshold is raised to $T=0.60$ and correlation is reduced uniformly.",
            "tab:p4noise", "lccc", place="[H]")

# ---------------- figures ----------------
# index curves and expected responders
from matplotlib.lines import Line2D
dd = np.linspace(4, 300, 300)
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
for ct in CTYPES:
    axes[0].plot(dd, index(ct, dd), color=CALLS[ct]["color"], label=CALLS[ct]["label"])
    axes[0].plot(dd, fidelity(ct, dd), color=CALLS[ct]["color"], lw=0.7, ls=":")
    axes[0].axvline(radius(ct), color=CALLS[ct]["color"], lw=0.6, ls="--", alpha=0.7)
    axes[1].plot(dd, N_TYP * PHI * pred_p(ct, dd), color=CALLS[ct]["color"])
axes[0].axhline(ISTAR, color="grey", lw=0.7, ls="--")
axes[0].text(220, ISTAR + 0.02, "$I^{\\ast}=0.5$", fontsize=7, color="grey")
axes[0].set_xlabel("Distance (m)"); axes[0].set_ylabel("Effectiveness index $I$")
axes[0].set_ylim(0, 1)
h, l = axes[0].get_legend_handles_labels()
h.append(Line2D([0], [0], color="grey", ls=":", lw=0.8, label="Fidelity $F$ (dotted)"))
axes[0].legend(handles=h, fontsize=6.5, loc="lower left")
axes[1].axhline(1, color="grey", lw=0.7, ls="--")
axes[1].set_xlabel("Distance (m)"); axes[1].set_ylabel("Expected responders")
axes[1].set_ylim(0, 5)
fig.tight_layout()
save(fig, "fig_p4_index")

# radii under scenarios
fig, ax = plt.subplots(figsize=(3.6, 2.6))
scen = [("Baseline", 0.0, None), ("$-10\\%$", 0.10, None), ("$-20\\%$", 0.20, None),
        ("$T=0.6$", 0.0, 0.60), ("$T=0.7$", 0.0, 0.70)]
w = 0.15
for i, (lab, delta, tcap) in enumerate(scen):
    vals = [radius(ct, delta=delta, tcap=tcap) for ct in CTYPES]
    ax.bar(np.arange(3) + (i - 2) * w, vals, width=w, label=lab, edgecolor="none")
ax.set_xticks(range(3)); ax.set_xticklabels(["Flat", "Mod.", "2-voice"])
ax.set_ylabel("Radius (m)"); ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
save(fig, "fig_p4_radii")

# radius vs correlation loss for each threshold
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), sharey=True)
deltas = np.linspace(0, 0.2, 9)
for ax, ct in zip(axes, CTYPES):
    for t, ls in zip([0.50, 0.60, 0.70], ["-", "--", ":"]):
        vals = [radius(ct, delta=de, tcap=t) for de in deltas]
        ax.plot(deltas * 100, vals, ls=ls, color=CALLS[ct]["color"], marker="o", ms=2.5,
                label=f"$T={t}$")
    ax.set_xlabel("Correlation loss (\\%)")
    ax.set_title(CALLS[ct]["label"], color=CALLS[ct]["color"])
    ax.set_ylim(0, 210)
axes[0].set_ylabel("Radius (m)"); axes[0].legend(fontsize=6)
fig.tight_layout()
save(fig, "fig_p4_heatmap")

dump("p4", dict(radius={ct: {str(k): v for k, v in rad_res[ct].items()} for ct in CTYPES},
                index_star=ISTAR, phi=PHI, group=N_TYP, weight_grid=w_res))
print(json.dumps(rad_res, indent=2))
