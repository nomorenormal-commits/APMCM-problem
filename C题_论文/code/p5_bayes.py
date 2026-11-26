"""Problem 2 upgrade: Bayesian hierarchical degradation model and probabilistic risk."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit
import emcee
import matplotlib.pyplot as plt

from common import (DIST, CTYPES, CALLS, MEASURES, load_degradation, logistic_decay,
                    decay_range, set_style, save, write_table, dump, DATA, RES)

set_style()
rng = np.random.RandomState(20261126)

# ------------------------------------------------------------------ data
long = load_degradation()
Y = {}
for ct in CTYPES:
    mat = np.full((300, 7), np.nan)
    vals = []
    for m in ["spec", "amp", "env"]:
        sub = long[(long.calltype == ct) & (long.measure == m)]
        M = np.full((300, 7), np.nan)
        for j, d in enumerate(DIST):
            v = sub.loc[sub.distance == d, "corr"].values
            M[:len(v), j] = v
        vals.append(M)
    z = np.arctanh(np.clip(np.stack(vals), 1e-4, 1 - 1e-6))
    Y[ct] = np.tanh(np.nanmean(z, axis=0))  # 300 x 7 composite replicates

# MLE starting values
P2 = json.loads((RES / "p2.json").read_text())
mle = {ct: np.array(P2["param"][ct]) for ct in CTYPES}


def hill(d, f0, fmin, dh, k):
    return fmin + (f0 - fmin) / (1 + (d / dh) ** k)


def logit(p):
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# parameter layout
# 0-2 a_k, 3-5 c_k, 6-8 log dh, 9-11 log k, 12-14 log sigma
# 15 mu_a, 16 log sd_a, 17 mu_c, 18 log sd_c,
# 19 mu_dh, 20 log sd_dh, 21 mu_k, 22 log sd_k
NDIM = 23


def unpack(theta):
    a = theta[0:3]
    c = theta[3:6]
    ldh = theta[6:9]
    lk = theta[9:12]
    lsig = theta[12:15]
    return a, c, ldh, lk, lsig


def log_prior(theta):
    a, c, ldh, lk, lsig = unpack(theta)
    mu_a, lsa, mu_c, lsc, mu_dh, lsdh, mu_k, lsk = theta[15:23]
    lp = 0.0
    lp += norm.logpdf(a, mu_a, np.exp(lsa)).sum()
    lp += norm.logpdf(c, mu_c, np.exp(lsc)).sum()
    lp += norm.logpdf(ldh, mu_dh, np.exp(lsdh)).sum()
    lp += norm.logpdf(lk, mu_k, np.exp(lsk)).sum()
    lp += norm.logpdf(lsig, np.log(0.10), 0.5).sum()
    # hyper-priors
    lp += norm.logpdf(mu_a, 1.5, 1.0)
    lp += norm.logpdf(lsa, np.log(0.4), 0.5)
    lp += norm.logpdf(mu_c, -1.5, 1.0)
    lp += norm.logpdf(lsc, np.log(0.5), 0.5)
    lp += norm.logpdf(mu_dh, np.log(120.0), 0.7)
    lp += norm.logpdf(lsdh, np.log(0.4), 0.5)
    lp += norm.logpdf(mu_k, np.log(2.5), 0.7)
    lp += norm.logpdf(lsk, np.log(0.5), 0.5)
    if not np.isfinite(lp):
        return -np.inf
    return lp


def log_like(theta):
    a, c, ldh, lk, lsig = unpack(theta)
    F0 = sigmoid(a)
    Fmin = F0 * sigmoid(c)
    dh = np.exp(ldh)
    kk = np.exp(lk)
    sig = np.exp(lsig)
    ll = 0.0
    for i, ct in enumerate(CTYPES):
        y = Y[ct]
        pred = Fmin[i] + (F0[i] - Fmin[i]) / (1 + (DIST[None, :] / dh[i]) ** kk[i])
        r = (y - pred) / sig[i]
        ok = np.isfinite(r)
        ll += -0.5 * np.sum(r[ok] ** 2) - np.sum(ok) * np.log(sig[i])
    return ll


def log_prob(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like(theta)


# starting point
p0 = np.zeros(NDIM)
for i, ct in enumerate(CTYPES):
    f0, fmin, dh, k = mle[ct]
    resid = Y[ct] - hill(DIST[None, :], f0, fmin, dh, k)
    p0[i] = logit(np.clip(f0, .01, .99))
    p0[3 + i] = logit(np.clip(fmin / f0, .01, .99))
    p0[6 + i] = np.log(dh)
    p0[9 + i] = np.log(k)
    p0[12 + i] = np.log(np.nanstd(resid))
p0[15] = p0[0:3].mean(); p0[16] = np.log(0.4)
p0[17] = p0[3:6].mean(); p0[18] = np.log(0.5)
p0[19] = p0[6:9].mean(); p0[20] = np.log(0.4)
p0[21] = p0[9:12].mean(); p0[22] = np.log(0.5)

nwalkers = 60
pos = p0 + 1e-2 * rng.randn(nwalkers, NDIM)
sampler = emcee.EnsembleSampler(nwalkers, NDIM, log_prob)
sampler.run_mcmc(pos, 2200, progress=False)
chain = sampler.get_chain(discard=700, thin=10, flat=True)
np.savez_compressed(RES / "p5_chain.npz", chain=chain[::10])
print("posterior samples", chain.shape, "acceptance", np.mean(sampler.acceptance_fraction))

# posterior summaries
post_rows = []
post = {}
for i, ct in enumerate(CTYPES):
    a = chain[:, i]; c = chain[:, 3 + i]; ldh = chain[:, 6 + i]
    lk = chain[:, 9 + i]; ls = chain[:, 12 + i]
    f0 = sigmoid(a); fmin = f0 * sigmoid(c); dh = np.exp(ldh); kk = np.exp(lk)
    sig = np.exp(ls)
    post[ct] = dict(f0=f0, fmin=fmin, dh=dh, k=kk, sigma=sig)
    def ci(x):
        return np.percentile(x, [2.5, 50, 97.5])
    post_rows.append([CALLS[ct]["label"],
                      f"{ci(f0)[1]:.3f} [{ci(f0)[0]:.3f}, {ci(f0)[2]:.3f}]",
                      f"{ci(fmin)[1]:.3f} [{ci(fmin)[0]:.3f}, {ci(fmin)[2]:.3f}]",
                      f"{ci(dh)[1]:.0f} [{ci(dh)[0]:.0f}, {ci(dh)[2]:.0f}]",
                      f"{ci(kk)[1]:.2f} [{ci(kk)[0]:.2f}, {ci(kk)[2]:.2f}]",
                      f"{ci(sig)[1]:.3f}"])
write_table("tab_p5_post", ["Call type", "$F_0$", "$F_{\\min}$", "$d_h$ (m)", "$\\kappa$", "$\\sigma$"],
            post_rows,
            "Posterior medians and $95\\%$ credible intervals of the Bayesian hierarchical Hill model.",
            "tab:p5post", "lccccc", size="\\footnotesize")

# ------------------------------------------------------------------ response model with uncertainty
import statsmodels.formula.api as smf
import patsy

d = pd.read_csv(DATA / "Data02_AfricanPenguins_playbacks_data.csv")
d["CallType"] = pd.Categorical(d["CallType"], categories=["control", "flat", "mod", "tv"], ordered=True)
d["Context"] = pd.Categorical(d["Context"], categories=["commuting", "searching", "after"], ordered=True)
d["log2D"] = np.log2(d["Distance"]); d["log2N"] = np.log2(d["Npenguins"])
resp = smf.logit("ResponseYesNo ~ C(CallType, Treatment('control')) + C(Context, Treatment('commuting')) + log2D + log2N",
                 data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["expID"]}, maxiter=500)
DI = resp.model.data.orig_exog.design_info
beta_draws = rng.multivariate_normal(resp.params, resp.cov_params(), size=len(chain))

# empirical group-size distribution
Npool = d["Npenguins"].values

# model-form parameters (fit to means) and AIC weights
def m_exp(dd, f0, lam): return f0 * np.exp(-lam * dd)
def m_pow(dd, f0, d0, k): return f0 * (1 + dd / d0) ** (-k)
def m_wei(dd, fmin, f0, lam, k): return fmin + (f0 - fmin) * np.exp(-(dd / lam) ** k)
FAM = {
    "hill": (logistic_decay, [0.8, .05, 100, 2], ([0, 0, .5, .1], [1.05, .6, 5000, 15])),
    "weibull": (m_wei, [.05, .8, 100, 1.2], ([0, 0, .5, .1], [1.05, 1.05, 5000, 15])),
    "exp": (m_exp, [.8, .01], ([0, 0], [1.05, .2])),
    "power": (m_pow, [.8, 30, .7], ([0, .1, .05], [1.05, 5000, 6])),
}
fam_par = {}
fam_aic = {}
for ct in CTYPES:
    comp = np.nanmean(Y[ct], axis=0)
    fam_par[ct] = {}; fam_aic[ct] = {}
    for name, (fn, p0f, bnds) in FAM.items():
        try:
            p, _ = curve_fit(fn, DIST, comp, p0=p0f, bounds=bnds, maxfev=200000)
            fam_par[ct][name] = p
            rss = np.sum((comp - fn(DIST, *p)) ** 2)
            fam_aic[ct][name] = 7 * np.log(max(rss, 1e-12) / 7) + 2 * len(p)
        except Exception:
            fam_aic[ct][name] = np.inf

# ------------------------------------------------------------------ probabilistic risk Monte Carlo
grid = np.linspace(4, 350, 347)
# posterior draw indices
idx = rng.choice(len(chain), size=min(400, len(chain)), replace=False)
risk = {ct: [] for ct in CTYPES}
radius_surv = {ct: np.zeros_like(grid) for ct in CTYPES}
resp100 = {ct: [] for ct in CTYPES}
resp150 = {ct: [] for ct in CTYPES}

for s_i, s in enumerate(idx):
    # draw response coefficients
    b = beta_draws[s_i]
    N = rng.choice(Npool)
    # threshold, noise, responder fraction
    T = 0.45 + 0.30 * rng.beta(4, 4)
    delta = 0.20 * rng.beta(1.5, 4.0)
    phi = 0.20 + 0.60 * rng.beta(2, 2)
    for ct in CTYPES:
        # fidelity from posterior Hill
        f0 = sigmoid(chain[s, CTYPES.index(ct)])
        cc = sigmoid(chain[s, 3 + CTYPES.index(ct)])
        fmin = f0 * cc
        dh = np.exp(chain[s, 6 + CTYPES.index(ct)])
        kk = np.exp(chain[s, 9 + CTYPES.index(ct)])
        F = (1 - delta) * hill(grid, f0, fmin, dh, kk)
        # response probability from sampled coefficients
        nd = pd.DataFrame({"CallType": ct, "Context": "searching", "Distance": grid,
                           "log2D": np.log2(grid), "log2N": np.log2(N), "Npenguins": N})
        X = patsy.build_design_matrices([DI], nd, return_type="dataframe")[0].values
        P = 1 / (1 + np.exp(-(X @ b)))
        I = np.sqrt(np.clip(F, 0, 1) * P)
        ok = (F >= T) & (I >= 0.5)
        rad = grid[ok].max() if ok.any() else 0.0
        risk[ct].append(rad)
        radius_surv[ct] += (rad >= grid).astype(float)
        resp100[ct].append(N * phi * P[np.argmin(abs(grid - 100))])
        resp150[ct].append(N * phi * P[np.argmin(abs(grid - 150))])

for ct in CTYPES:
    radius_surv[ct] /= len(idx)

risk_rows = []
risk_summary = {}
for ct in CTYPES:
    r = np.array(risk[ct])
    risk_summary[ct] = dict(median=float(np.median(r)),
                            lo=float(np.percentile(r, 2.5)),
                            hi=float(np.percentile(r, 97.5)),
                            p100=float(np.mean(r >= 100)), p120=float(np.mean(r >= 120)),
                            p150=float(np.mean(r >= 150)), p50=float(np.mean(r < 50)),
                            e100=float(np.median(resp100[ct])),
                            e100lo=float(np.percentile(resp100[ct], 2.5)),
                            e100hi=float(np.percentile(resp100[ct], 97.5)),
                            e150=float(np.median(resp150[ct])),
                            e150lo=float(np.percentile(resp150[ct], 2.5)),
                            e150hi=float(np.percentile(resp150[ct], 97.5)))
    risk_rows.append([CALLS[ct]["label"],
                      f"{np.median(r):.0f} [{np.percentile(r,2.5):.0f}, {np.percentile(r,97.5):.0f}]",
                      f"{100*np.mean(r>=100):.0f}", f"{100*np.mean(r>=120):.0f}",
                      f"{100*np.mean(r>=150):.0f}", f"{100*np.mean(r<50):.0f}"])
write_table("tab_p5_risk", ["Call type", "Radius (m)", "$P(r\\ge100)$", "$P(r\\ge120)$",
                            "$P(r\\ge150)$", "$P(r<50)$"], risk_rows,
            "Probabilistic effective radius integrating posterior, threshold, noise, responder-fraction and group-size uncertainty.",
            "tab:p5risk", "lccccc", size="\\footnotesize")

# ------------------------------------------------------------------ extrapolation beyond 256 m
extra_rows = []
for ct in CTYPES:
    row = [CALLS[ct]["label"]]
    for name in ["hill", "weibull", "exp", "power"]:
        p = fam_par[ct][name]
        row.append(f"{FAM[name][0](300, *p):.2f} / {FAM[name][0](500, *p):.2f}")
    extra_rows.append(row)
write_table("tab_p5_extra", ["Call type", "Hill", "Weibull", "Exponential", "Power"], extra_rows,
            "Predicted fidelity at 300 m and 500 m by model family, illustrating extrapolation uncertainty beyond the observed 256 m.",
            "tab:p5extra", "lcccc")

# ------------------------------------------------------------------ figures
# posterior fan chart
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), sharey=True)
xx = np.linspace(4, 256, 250)
for ax, ct in zip(axes, CTYPES):
    i = CTYPES.index(ct)
    f0 = sigmoid(chain[:, i]); cc = sigmoid(chain[:, 3 + i])
    fmin = f0 * cc; dh = np.exp(chain[:, 6 + i]); kk = np.exp(chain[:, 9 + i])
    band = np.array([hill(xx, f0[j], fmin[j], dh[j], kk[j]) for j in range(0, len(chain), 8)])
    lo, mid, hi = np.percentile(band, [5, 50, 95], axis=0)
    ax.fill_between(xx, lo, hi, color=CALLS[ct]["color"], alpha=.18, lw=0)
    ax.plot(xx, mid, color=CALLS[ct]["color"])
    ax.plot(DIST, np.nanmean(Y[ct], axis=0), "o", color=CALLS[ct]["color"], ms=3)
    ax.set_xscale("log", base=2); ax.set_xticks(DIST); ax.set_xticklabels([str(int(x)) for x in DIST], fontsize=7)
    ax.set_xlabel("Distance (m)"); ax.set_title(CALLS[ct]["label"], color=CALLS[ct]["color"])
    ax.set_ylim(0, 1)
axes[0].set_ylabel("Composite fidelity $F(d)$")
fig.tight_layout(); save(fig, "fig_p5_posterior")

# radius survival curves
fig, ax = plt.subplots(figsize=(4.3, 2.8))
for ct in CTYPES:
    ax.plot(grid, radius_surv[ct], color=CALLS[ct]["color"], label=CALLS[ct]["label"])
ax.axvline(120, color="grey", lw=.7, ls="--")
ax.text(122, .95, "120 m", fontsize=7, color="grey")
ax.set_xlabel("Radius threshold $r$ (m)"); ax.set_ylabel("$P(R\\geq r)$")
ax.set_ylim(0, 1); ax.legend(fontsize=7)
fig.tight_layout(); save(fig, "fig_p5_radius")

# extrapolation
fig, ax = plt.subplots(figsize=(4.3, 2.8))
xx2 = np.linspace(4, 500, 300)
cols = {"hill": "#1f6fb2", "weibull": "#d17a22", "exp": "#7a7a7a", "power": "#2e7d5b"}
for name in ["hill", "weibull", "exp", "power"]:
    p = fam_par["tv"][name]
    ax.plot(xx2, FAM[name][0](xx2, *p), color=cols[name], label=name)
ax.axvline(256, color="grey", lw=.7, ls=":")
ax.set_xlabel("Distance (m)"); ax.set_ylabel("Fidelity $F$")
ax.set_xlim(4, 500); ax.set_ylim(0, 1); ax.legend(fontsize=7)
ax.set_title("Two-voice extrapolation", fontsize=9)
fig.tight_layout(); save(fig, "fig_p5_extrap")

dump("p5", dict(posterior={ct: {k: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
                                 for k, v in post[ct].items()} for ct in CTYPES},
                risk=risk_summary,
                fam_aic={ct: {k: float(v) for k, v in fam_aic[ct].items()} for ct in CTYPES}))
print(json.dumps(risk_summary, indent=2))
