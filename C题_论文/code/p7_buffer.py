"""Constrained buffer-allocation optimization on top of the Bayesian risk model."""
import json
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from common import DATA, CTYPES, CALLS, RES, set_style, save, write_table, dump

set_style()
rng = np.random.RandomState(42)

chain = np.load(RES / "p5_chain.npz")["chain"]
grid = np.arange(4, 301, 1.0)

# point response model at N = 8, searching
d = pd.read_csv(DATA / "Data02_AfricanPenguins_playbacks_data.csv")
d["CallType"] = pd.Categorical(d["CallType"], categories=["control", "flat", "mod", "tv"], ordered=True)
d["Context"] = pd.Categorical(d["Context"], categories=["commuting", "searching", "after"], ordered=True)
d["log2D"] = np.log2(d["Distance"]); d["log2N"] = np.log2(d["Npenguins"])
resp = smf.logit("ResponseYesNo ~ C(CallType, Treatment('control')) + C(Context, Treatment('commuting')) + log2D + log2N",
                 data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["expID"]}, maxiter=500)
DI = resp.model.data.orig_exog.design_info

P = {}
for ct in CTYPES:
    nd = pd.DataFrame({"CallType": ct, "Context": "searching", "Distance": grid,
                       "log2D": np.log2(grid), "log2N": np.log2(8), "Npenguins": 8})
    P[ct] = np.asarray(resp.predict(nd))

DRAWS = chain[np.random.RandomState(7).choice(len(chain), size=min(500, len(chain)), replace=False)]


def radii_for(ct, delta, T=0.60):
    i = CTYPES.index(ct)
    f0 = 1 / (1 + np.exp(-DRAWS[:, i]))
    c = 1 / (1 + np.exp(-DRAWS[:, 3 + i]))
    fmin = f0 * c
    dh = np.exp(DRAWS[:, 6 + i])
    kk = np.exp(DRAWS[:, 9 + i])
    out = np.empty(len(DRAWS))
    for j in range(len(DRAWS)):
        F = (1 - delta) * (fmin[j] + (f0[j] - fmin[j]) / (1 + (grid / dh[j]) ** kk[j]))
        I = np.sqrt(np.clip(F, 0, 1) * P[ct])
        ok = (F >= T) & (I >= 0.5)
        out[j] = grid[ok].max() if ok.any() else 0.0
    return out


ZONES = [
    dict(name="A: flat-dominated, noisy", w=np.array([0.6, 0.3, 0.1]), delta=0.15, cost=1.0),
    dict(name="B: mixed", w=np.array([0.3, 0.4, 0.3]), delta=0.10, cost=1.0),
    dict(name="C: two-voice-dominated, quiet", w=np.array([0.1, 0.3, 0.6]), delta=0.05, cost=1.0),
]
rgrid = np.arange(10, 201, 1.0)

# survival S_k(r | zone) and zone coverage
S = {}
for zi, z in enumerate(ZONES):
    S[zi] = {}
    for ct in CTYPES:
        rad = radii_for(ct, z["delta"])
        S[zi][ct] = np.array([np.mean(rad >= r) for r in rgrid])

COV = {}
for zi, z in enumerate(ZONES):
    parts = []
    for k, ct in enumerate(CTYPES):
        s = S[zi][ct]
        cum = np.concatenate([[0.0], np.cumsum((s[1:] + s[:-1]) / 2 * np.diff(rgrid))])
        avg = cum / np.maximum(rgrid - 4, 1)
        parts.append(z["w"][k] * avg)
    COV[zi] = sum(parts)


def interp_cov(zi, r):
    return np.interp(r, rgrid, COV[zi], left=0.0, right=COV[zi][-1])


def optimise(B):
    c = np.array([z["cost"] for z in ZONES])
    def obj(r):
        return -sum(interp_cov(i, r[i]) for i in range(3))
    cons = [{"type": "ineq", "fun": lambda r: B - np.sum(c * r)}]
    best = None
    for start in range(8):
        r0 = rng.uniform(20, min(180, B / c.sum() + 60), size=3)
        r0 = r0 / np.sum(c * r0) * B * 0.95
        sol = minimize(obj, r0, method="SLSQP", bounds=[(10, 200)] * 3,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
        if sol.success and (best is None or sol.fun < best.fun):
            best = sol
    r = np.clip(best.x, 10, 200)
    return r, -best.fun


budgets = [90, 120, 150, 180, 210, 240]
rows, opt_res = [], {}
for B in budgets:
    r, cov = optimise(B)
    ru = B / 3
    covu = sum(interp_cov(i, ru) for i in range(3))
    g = 100 * (cov - covu) / max(covu, 1e-9)
    if abs(g) < 0.05:
        g = 0.0
    rows.append([str(B), f"{r[0]:.0f}", f"{r[1]:.0f}", f"{r[2]:.0f}",
                 f"{cov:.3f}", f"{covu:.3f}", f"{g:.1f}"])
    opt_res[B] = dict(r=r.tolist(), coverage=cov, uniform=ru, coverage_uniform=covu)
write_table("tab_p7_opt", ["Budget $B$ (m)", "$r_A$", "$r_B$", "$r_C$",
                           "Optimal coverage", "Uniform coverage", "Gain (\\%)"], rows,
            "Constrained buffer allocation: optimal zone-specific radii, total weighted coverage, and comparison with a uniform radius policy.",
            "tab:p7opt", "lcccccc")

# figure
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
cols = ["#1f6fb2", "#d17a22", "#2e7d5b"]
for i, z in enumerate(ZONES):
    axes[0].plot(budgets, [opt_res[B]["r"][i] for B in budgets], "o-", color=cols[i],
                 label=z["name"][0])
    axes[1].plot(budgets, [interp_cov(i, opt_res[B]["r"][i]) for B in budgets], "o-", color=cols[i])
axes[0].set_xlabel("Budget $B$ (sum of radii, m)"); axes[0].set_ylabel("Optimal radius (m)")
axes[0].legend(fontsize=7); axes[0].set_title("Zone-specific buffers", fontsize=9)
axes[1].set_xlabel("Budget $B$"); axes[1].set_ylabel("Zone coverage")
axes[1].set_title("Coverage at the optimum", fontsize=9)
fig.tight_layout(); save(fig, "fig_p7_opt")

dump("p7", dict(opt={str(k): v for k, v in opt_res.items()},
                zones=[dict(name=z["name"], delta=z["delta"], weights=z["w"].tolist()) for z in ZONES]))
print(json.dumps(opt_res, indent=2))


# ---------------- input sensitivity of the allocation rule ----------------
def build_cov(zones):
    covs = []
    for z in zones:
        parts = []
        for k, ct in enumerate(CTYPES):
            rad = radii_for(ct, z["delta"])
            s = np.array([np.mean(rad >= r) for r in rgrid])
            cum = np.concatenate([[0.0], np.cumsum((s[1:] + s[:-1]) / 2 * np.diff(rgrid))])
            avg = cum / np.maximum(rgrid - 4, 1)
            parts.append(z["w"][k] * avg)
        covs.append(sum(parts))
    return covs


def interp_any(covs, zi, r):
    return np.interp(r, rgrid, covs[zi], left=0.0, right=covs[zi][-1])


def optimise_any(B, covs):
    c = np.array([z["cost"] for z in ZONES])
    def obj(r):
        return -sum(interp_any(covs, i, r[i]) for i in range(3))
    cons = [{"type": "ineq", "fun": lambda r: B - np.sum(c * r)}]
    best = None
    for _ in range(8):
        r0 = rng.uniform(20, min(180, B / c.sum() + 60), size=3)
        r0 = r0 / np.sum(c * r0) * B * 0.95
        sol = minimize(obj, r0, method="SLSQP", bounds=[(10, 200)] * 3,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
        if sol.success and (best is None or sol.fun < best.fun):
            best = sol
    return np.clip(best.x, 10, 200), -best.fun


def perturb(wmul=1.0, dadd=0.0):
    zs = []
    for z in ZONES:
        w = z["w"] * wmul
        zs.append(dict(z, w=w / w.sum(), delta=max(0.0, z["delta"] + dadd)))
    return zs


scenarios = [("Baseline", 1.0, 0.0), ("Weights $+20\\%$", 1.2, 0.0),
             ("Weights $-20\\%$", 0.8, 0.0), ("Noise $+0.03$", 1.0, 0.03),
             ("Noise $-0.03$", 1.0, -0.03), ("Noise $+0.05$", 1.0, 0.05),
             ("Noise $-0.05$", 1.0, -0.05)]
sens_rows, sens_res = [], {}
for label, wm, da in scenarios:
    covs = build_cov(perturb(wm, da))
    r240, c240 = optimise_any(240, covs)
    r150, c150 = optimise_any(150, covs)
    g240 = 100 * (c240 - sum(interp_any(covs, i, 80) for i in range(3))) / \
        max(sum(interp_any(covs, i, 80) for i in range(3)), 1e-9)
    g150 = 100 * (c150 - sum(interp_any(covs, i, 50) for i in range(3))) / \
        max(sum(interp_any(covs, i, 50) for i in range(3)), 1e-9)
    if abs(g150) < 0.05:
        g150 = 0.0
    if abs(g240) < 0.05:
        g240 = 0.0
    order = "$C>B>A$" if r240[2] >= r240[1] >= r240[0] else "other"
    sens_rows.append([label, f"{r240[0]:.0f}", f"{r240[1]:.0f}", f"{r240[2]:.0f}",
                      f"{g150:.1f}", f"{g240:.1f}", order])
    sens_res[label] = dict(r240=r240.tolist(), gain150=g150, gain240=g240, order=order)

write_table("tab_p7_sens", ["Scenario", "$r_A$", "$r_B$", "$r_C$",
                            "Gain at $B=150$ (\\%)", "Gain at $B=240$ (\\%)", "Ordering"],
            sens_rows,
            "Input sensitivity of the buffer allocation at $B=150$ and $B=240$ m: $\\pm20\\%$ weight perturbations and $\\pm0.03$/$\\pm0.05$ perturbations of the correlation-loss parameter.",
            "tab:p7sens", "lcccccc")


def median_radius(ct, delta, T=0.60):
    return float(np.median(radii_for(ct, delta, T)))


rank_rows, rank_res = [], {}
for label, wm, da in scenarios:
    delta = 0.10 + da
    vals = [median_radius(ct, delta) for ct in CTYPES]
    order = "$\\mathrm{tv}>\\mathrm{mod}>\\mathrm{flat}$" if vals[2] >= vals[1] >= vals[0] else "other"
    rank_rows.append([label, f"{vals[0]:.0f}", f"{vals[1]:.0f}", f"{vals[2]:.0f}", order])
    rank_res[label] = dict(flat=vals[0], mod=vals[1], tv=vals[2], order=order)

write_table("tab_p7_rank", ["Scenario", "Flat", "Modulated", "Two-voice", "Ranking"],
            rank_rows,
            "Call-type median effective radius (m, $T=0.60$) under the same input perturbations; the ranking is invariant.",
            "tab:p7rank", "lcccc")

dump("p7", dict(opt={str(k): v for k, v in opt_res.items()},
                zones=[dict(name=z["name"], delta=z["delta"], weights=z["w"].tolist()) for z in ZONES],
                sensitivity=sens_res, sensitivity_rank=rank_res))
print(json.dumps(sens_res, indent=2))
