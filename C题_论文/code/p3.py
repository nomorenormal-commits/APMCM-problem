"""Problem 3: playback response rates, combinations, latency and intensity models."""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.special import expit
import matplotlib.pyplot as plt

from common import CALLS, CTYPES, set_style, save, write_table, dump, fmt_p

set_style()
rng = np.random.RandomState(20261126)

d = pd.read_csv(__import__("common").DATA / "Data02_AfricanPenguins_playbacks_data.csv")
combo = []
for _, r in d.iterrows():
    s = "".join(c for c, v in zip("HVA", [r.HeadsUpYesNo, r.VocalYesNo, r.ApproachYesNo]) if v == 1)
    combo.append(s if s else "none")
d["Combo"] = combo
d["CallType"] = pd.Categorical(d["CallType"], categories=["control", "flat", "mod", "tv"], ordered=True)
d["Context"] = pd.Categorical(d["Context"], categories=["commuting", "searching", "after"], ordered=True)
d["log2D"] = np.log2(d["Distance"])
d["log2N"] = np.log2(d["Npenguins"])
d["Call"] = (d["Type"] == "call").astype(int)

# first overt response latency (only behaviours actually observed)
lat = d[["HeadsUpLatency", "VocalLatency", "ApproachLatency"]].copy()
lat.iloc[d["HeadsUpYesNo"].values == 0, 0] = np.nan
lat.iloc[d["VocalYesNo"].values == 0, 1] = np.nan
lat.iloc[d["ApproachYesNo"].values == 0, 2] = np.nan
d["FirstLatency"] = lat.min(axis=1)
d["Responded"] = d["ResponseYesNo"]

# ---------------- descriptive response rates ----------------
def rate_table(rows, mask, label):
    sub = d[mask]
    r = [label, len(sub)]
    for col in ["ResponseYesNo", "HeadsUpYesNo", "VocalYesNo", "ApproachYesNo"]:
        r.append(f"{sub[col].sum()}/{len(sub)} ({100*sub[col].mean():.1f}\\%)")
    rows.append(r)


desc_rows = []
rate_table(desc_rows, d["Type"] == "control", "Control")
for ct in CTYPES:
    rate_table(desc_rows, (d["Type"] == "call") & (d["CallType"] == ct), CALLS[ct]["label"])
rate_table(desc_rows, d["Type"] == "call", "All natural calls")
write_table("tab_p3_rates", ["Stimulus", "$N$", "Any response", "Head-up", "Vocal", "Approach"],
            desc_rows, "Observed response rates of playback experiments by stimulus type (count and percentage).",
            "tab:p3rates", "lrcccc")

# Fisher tests call vs control for each behaviour
fisher_rows = []
for col, lab in [("ResponseYesNo", "Any response"), ("HeadsUpYesNo", "Head-up"),
                 ("VocalYesNo", "Vocal"), ("ApproachYesNo", "Approach")]:
    tab = pd.crosstab(d["Type"], d[col]).reindex(["call", "control"])
    odds, p = stats.fisher_exact(tab.values)
    odds = 1.0 / odds if odds > 0 else np.inf
    fisher_rows.append([lab, f"{tab.loc['control',1]}/{tab.loc['control'].sum()}",
                        f"{tab.loc['call',1]}/{tab.loc['call'].sum()}",
                        (f"{odds:.2f}" if np.isfinite(odds) else "$\\infty$"), fmt_p(p)])
write_table("tab_p3_fisher", ["Behaviour", "Control", "Natural calls", "Odds ratio", "Fisher $p$"],
            fisher_rows, "Comparison of natural calls and control sounds (Fisher exact test, two-sided).",
            "tab:p3fisher", "lccrr", place="[H]")

# ---------------- logistic models ----------------
FORM = "~ C(CallType, Treatment('control')) + C(Context, Treatment('commuting')) + log2D + log2N"


def fit_logit(y, data=None):
    dd = d if data is None else data
    return smf.logit(f"{y} {FORM}", data=dd).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": dd["expID"]}, maxiter=500)


models = {}
for y in ["ResponseYesNo", "HeadsUpYesNo", "VocalYesNo"]:
    models[y] = fit_logit(y)

# approach: control has zero events, so fit natural calls only
dcall = d[d.Type == "call"].copy()
dcall["CallType"] = dcall["CallType"].cat.remove_unused_categories()
models["ApproachYesNo"] = smf.logit(
    "ApproachYesNo ~ C(CallType, Treatment('flat')) + C(Context, Treatment('commuting')) + log2D + log2N",
    data=dcall).fit(disp=0, cov_type="cluster", cov_kwds={"groups": dcall["expID"]}, maxiter=500)

or_rows = []
terms = [("C(CallType, Treatment('control'))[T.flat]", "Flat vs control"),
         ("C(CallType, Treatment('control'))[T.mod]", "Modulated vs control"),
         ("C(CallType, Treatment('control'))[T.tv]", "Two-voice vs control"),
         ("C(Context, Treatment('commuting'))[T.searching]", "Searching vs commuting"),
         ("C(Context, Treatment('commuting'))[T.after]", "After feeding vs commuting"),
         ("log2D", "Distance (per doubling)"),
         ("log2N", "Group size (per doubling)")]
for y, lab in [("ResponseYesNo", "Any response"), ("HeadsUpYesNo", "Head-up"),
               ("VocalYesNo", "Vocal")]:
    res = models[y]
    for term, tlab in terms:
        if term in res.params.index:
            b, se = res.params[term], res.bse[term]
            or_rows.append([lab, tlab, f"{np.exp(b):.2f}", f"{np.exp(b-1.96*se):.2f}--{np.exp(b+1.96*se):.2f}",
                            fmt_p(res.pvalues[term])])
# approach table uses flat baseline
for term, tlab in [("C(CallType, Treatment('flat'))[T.mod]", "Modulated vs flat"),
                   ("C(CallType, Treatment('flat'))[T.tv]", "Two-voice vs flat"),
                   ("C(Context, Treatment('commuting'))[T.searching]", "Searching vs commuting"),
                   ("C(Context, Treatment('commuting'))[T.after]", "After feeding vs commuting"),
                   ("log2D", "Distance (per doubling)"),
                   ("log2N", "Group size (per doubling)")]:
    res = models["ApproachYesNo"]
    b, se = res.params[term], res.bse[term]
    or_rows.append(["Approach (calls only)", tlab, f"{np.exp(b):.2f}",
                    f"{np.exp(b-1.96*se):.2f}--{np.exp(b+1.96*se):.2f}", fmt_p(res.pvalues[term])])
write_table("tab_p3_or", ["Outcome", "Contrast", "OR", "95\\% CI", "$p$"], or_rows,
            "Cluster-robust logistic regression of response behaviours on call type, context, distance and group size.",
            "tab:p3or", "llccc", size="\\footnotesize")

# ---------------- response combinations ----------------
combo_order = ["none", "H", "V", "A", "HV", "HA", "VA", "HVA"]
combo_tab = []
for combo_name in combo_order:
    sub = d[d.Combo == combo_name]
    row = [combo_name.replace("none", "None")] + [str(len(sub[sub.Type == "control"])),
                                                  str(len(sub[sub.Type == "call"]))]
    combo_tab.append(row)
write_table("tab_p3_combo", ["Combination", "Control", "Natural calls"], combo_tab,
            "Frequencies of the eight response-behaviour combinations (H = head-up, V = vocal, A = approach).",
            "tab:p3combo", "lcc", place="[H]")

# collapsed categories for the multinomial model
def collapse(c):
    if c == "none":
        return "none"
    if c == "H":
        return "H"
    if c == "V":
        return "V"
    if c == "A":
        return "A"
    return "multi"


d["Cat"] = d.Combo.map(collapse)
sub = d[d.Cat != "A"].copy()  # a single approach-only case
cat_order = ["none", "H", "V", "multi"]
sub["Cat"] = pd.Categorical(sub.Cat, categories=cat_order, ordered=True)
Xm = patsy.dmatrix("C(CallType, Treatment('control')) + C(Context, Treatment('commuting')) + log2D + log2N",
                   sub, return_type="dataframe")
mn = sm.MNLogit(sub["Cat"].cat.codes, Xm).fit(
    cov_type="cluster", cov_kwds={"groups": sub["expID"]}, maxiter=2000, disp=0)

# predicted distributions under representative conditions
def pred_mn(calltype, distance, context="searching", n=8):
    nd = pd.DataFrame({"CallType": [calltype], "Context": [context], "Distance": [distance],
                       "log2D": [np.log2(distance)], "log2N": [np.log2(n)], "Npenguins": [n]})
    X = patsy.build_design_matrices([Xm.design_info], nd, return_type="dataframe")[0]
    return mn.predict(X).values[0]


pred_rows = []
for ct in ["control", "flat", "mod", "tv"]:
    for dist in [50, 100, 150]:
        p = pred_mn(ct, dist)
        pred_rows.append([CALLS[ct]["label"], str(dist)] + [f"{100*v:.1f}" for v in p])
write_table("tab_p3_combo_pred", ["Call type", "Distance (m)", "None", "Head-up only",
                                  "Vocal only", "Multi-modal"], pred_rows,
            "Predicted probability (\\%) of each response combination at the median group size ($N=8$) in the searching context.",
            "tab:p3combopred", "llcccc", place="[H]")

# permutation test of call-type effect on combination distribution
obs = pd.crosstab(d["CallType"], d["Cat"])
chi2_obs = stats.chi2_contingency(obs.values)[0]
perm = []
for _ in range(2000):
    shuffled = rng.permutation(d["Cat"].values)
    perm.append(stats.chi2_contingency(pd.crosstab(d["CallType"], shuffled).values)[0])
p_perm = (np.sum(np.array(perm) >= chi2_obs) + 1) / (len(perm) + 1)

# ---------------- two-part latency model ----------------
# part 1: any response; part 2: log(first latency + 0.5) among responders
p1 = models["ResponseYesNo"]
resp = d[d.Responded == 1].copy()
resp["logLat"] = np.log(resp["FirstLatency"] + 0.5)
p2 = smf.ols("logLat ~ C(CallType, Treatment('control')) + C(Context, Treatment('commuting')) + log2D + log2N",
             data=resp).fit(cov_type="cluster", cov_kwds={"groups": resp["expID"]})
sigma = np.sqrt(p2.scale)

NSIM = 4000
def sim_design(res, nd):
    di = res.model.data.orig_exog.design_info
    return patsy.build_design_matrices([di], nd, return_type="dataframe")[0]


def simulate_logit(res, nd):
    X = sim_design(res, nd)
    beta = rng.multivariate_normal(res.params, res.cov_params(), NSIM)
    return expit(X.values @ beta.T)


def simulate_ols(res, nd):
    X = sim_design(res, nd)
    beta = rng.multivariate_normal(res.params, res.cov_params(), NSIM)
    return X.values @ beta.T


lat_rows = []
lat_res = {}
for ct in ["control", "flat", "mod", "tv"]:
    for dist in [50, 100, 150]:
        nd = pd.DataFrame({"CallType": [ct], "Context": ["searching"], "Distance": [dist],
                           "log2D": [np.log2(dist)], "log2N": [np.log2(8)], "Npenguins": [8]})
        p = simulate_logit(p1, nd)[0]
        mu = simulate_ols(p2, nd)[0]
        lat_sim = np.exp(mu + rng.normal(0, sigma, NSIM))
        # conditional median latency
        med = np.exp(mu)
        # unconditional expected latency
        uncond = p * np.exp(mu + sigma ** 2 / 2)
        lat_rows.append([CALLS[ct]["label"], str(dist),
                         f"{100*p.mean():.1f} [{100*np.percentile(p,2.5):.1f}, {100*np.percentile(p,97.5):.1f}]",
                         f"{np.median(med):.1f} [{np.percentile(med,2.5):.1f}, {np.percentile(med,97.5):.1f}]",
                         f"{uncond.mean():.1f} [{np.percentile(uncond,2.5):.1f}, {np.percentile(uncond,97.5):.1f}]"])
        lat_res[(ct, dist)] = dict(p=float(p.mean()), p_lo=float(np.percentile(p, 2.5)),
                                   p_hi=float(np.percentile(p, 97.5)),
                                   med=float(np.median(med)), med_lo=float(np.percentile(med, 2.5)),
                                   med_hi=float(np.percentile(med, 97.5)),
                                   exp=float(uncond.mean()), exp_lo=float(np.percentile(uncond, 2.5)),
                                   exp_hi=float(np.percentile(uncond, 97.5)))
write_table("tab_p3_latency", ["Call type", "Distance (m)", "P(response) (\\%)",
                               "Median latency $|$ response (s)", "Expected latency (s)"], lat_rows,
            "Two-part latency predictions at the median group size ($N=8$) in the searching context; brackets are 95\\% simulation intervals from the cluster-robust covariance.",
            "tab:p3lat", "llccc", size="\\footnotesize")

# ---------------- intensity models ----------------
def hurdle_predictions(outcome):
    pos = d[d[outcome] > 0].copy()
    pos["logY"] = np.log(pos[outcome])
    d["_pos"] = (d[outcome] > 0).astype(int)
    occ = smf.logit("_pos ~ C(CallType, Treatment('control')) + C(Context, Treatment('commuting')) + log2D + log2N",
                    data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["expID"]}, maxiter=500)
    mag = smf.ols("logY ~ C(CallType, Treatment('control')) + C(Context, Treatment('commuting')) + log2D + log2N",
                  data=pos).fit(cov_type="cluster", cov_kwds={"groups": pos["expID"]})
    return occ, mag, np.sqrt(mag.scale)


int_rows = []
for outcome, lab in [("HeadsUpDuration", "Head-up duration (s)"),
                     ("VocalNcalls", "Vocal count"), ("ApproachDuration", "Approach duration (s)")]:
    occ, mag, sg = hurdle_predictions(outcome)
    for ct in CTYPES:
        nd = pd.DataFrame({"CallType": [ct], "Context": ["searching"], "Distance": [100],
                           "log2D": [np.log2(100)], "log2N": [np.log2(8)], "Npenguins": [8]})
        po = simulate_logit(occ, nd)[0]
        mu = simulate_ols(mag, nd)[0]
        val = po * np.exp(mu + sg ** 2 / 2) if outcome == "VocalNcalls" else po * np.exp(mu + sg ** 2 / 2)
        int_rows.append([lab, CALLS[ct]["label"], f"{100*po.mean():.1f}",
                         f"{np.median(np.exp(mu)):.1f}", f"{np.median(val):.1f}"])
write_table("tab_p3_intensity", ["Outcome", "Call type", "P($>0$) (\\%)", "Median $|>0$",
                                 "Expected value"], int_rows,
            "Hurdle models for response-intensity measures at 100 m, $N=8$, searching context.",
            "tab:p3int", "llccc", place="[H]")

# ---------------- figures ----------------
# response rates by context and call type
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), sharey=True)
width = 0.2
for ax, (col, lab) in zip(axes, [("HeadsUpYesNo", "Head-up"), ("VocalYesNo", "Vocal"),
                                 ("ApproachYesNo", "Approach")]):
    for k, ct in enumerate(["control", "flat", "mod", "tv"]):
        vals = [100 * d[(d.CallType == ct) & (d.Context == cx)][col].mean()
                for cx in ["commuting", "searching", "after"]]
        ax.bar(np.arange(3) + (k - 1.5) * width, vals, width=width,
               color=CALLS[ct]["color"], label=CALLS[ct]["label"], edgecolor="none")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Commut.", "Search.", "After"])
    ax.set_title(lab); ax.set_ylim(0, 100)
axes[0].set_ylabel("Response rate (\\%)")
axes[0].legend(fontsize=6, loc="upper left")
fig.tight_layout()
save(fig, "fig_p3_rates")

# combination stacked bars
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
combo_labels = {"none": "None", "H": "H", "V": "V", "A": "A", "HV": "H+V", "HA": "H+A",
                "VA": "V+A", "HVA": "H+V+A"}
combo_colors = {"none": "#d9d9d9", "H": "#4e79a7", "V": "#59a14f", "A": "#f28e2b",
                "HV": "#76b7b2", "HA": "#e15759", "VA": "#b07aa1", "HVA": "#edc948"}

ax = axes[0]
labels = ["Control"] + [CALLS[c]["label"] for c in CTYPES]
keys = ["control"] + CTYPES
bottom = np.zeros(4)
for c in combo_order:
    vals = np.array([100 * (d[d.CallType == k].Combo == c).mean() for k in keys])
    ax.bar(range(4), vals, bottom=bottom, color=combo_colors[c], edgecolor="white",
           lw=0.5, width=0.62, label=combo_labels[c])
    bottom += vals
ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("Share of experiments (\\%)"); ax.set_ylim(0, 100)

ax = axes[1]
ctxs = ["commuting", "searching", "after"]
sub = d[d.Type == "call"]
bottom = np.zeros(3)
for c in combo_order:
    vals = np.array([100 * (sub[sub.Context == cx].Combo == c).mean() for cx in ctxs])
    ax.bar(range(3), vals, bottom=bottom, color=combo_colors[c], edgecolor="white",
           lw=0.5, width=0.6)
    bottom += vals
ax.set_xticks(range(3)); ax.set_xticklabels(["Commuting", "Searching", "After"])
ax.set_ylabel("Share of call experiments (\\%)"); ax.set_ylim(0, 100)
handles, labs = axes[0].get_legend_handles_labels()
fig.legend(handles, labs, ncol=8, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, -0.06),
           frameon=False)
fig.tight_layout()
save(fig, "fig_p3_combos")

# response probability and latency vs distance
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
dd = np.linspace(5, 200, 100)
for ct in ["control", "flat", "mod", "tv"]:
    nd = pd.DataFrame({"CallType": ct, "Context": "searching", "Distance": dd,
                       "log2D": np.log2(dd), "log2N": np.log2(8), "Npenguins": 8})
    p = simulate_logit(p1, nd)
    axes[0].plot(dd, 100 * p.mean(axis=1), color=CALLS[ct]["color"], label=CALLS[ct]["label"])
    mu = simulate_ols(p2, nd)
    axes[1].plot(dd, np.exp(mu).mean(axis=1), color=CALLS[ct]["color"])
axes[0].set_xlabel("Distance (m)"); axes[0].set_ylabel("P(response) (\\%)")
axes[0].legend(fontsize=7)
axes[1].set_xlabel("Distance (m)"); axes[1].set_ylabel("Median latency (s)")
fig.tight_layout()
save(fig, "fig_p3_latency")

dump("p3", dict(term_perm_p=float(p_perm), chi2=float(chi2_obs),
                latency={f"{k[0]}_{k[1]}": v for k, v in lat_res.items()}))
print("done")
