"""Causal sensitivity and response-cascade upgrade for the playback analysis."""
import json
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
import matplotlib.pyplot as plt

from common import DATA, CTYPES, CALLS, set_style, save, write_table, dump

set_style()
rng = np.random.RandomState(20261126)

d = pd.read_csv(DATA / "Data02_AfricanPenguins_playbacks_data.csv")
d["CallType"] = pd.Categorical(d["CallType"], categories=["control", "flat", "mod", "tv"], ordered=True)
d["Context"] = pd.Categorical(d["Context"], categories=["commuting", "searching", "after"], ordered=True)
d["log2D"] = np.log2(d["Distance"]); d["log2N"] = np.log2(d["Npenguins"])

# ------------------------------------------------------------------ E-values
def e_value(OR):
    """VanderWeele E-value for an odds ratio, common-outcome approximation RR=sqrt(OR)."""
    if OR == 1:
        return 1.0
    rr = np.sqrt(OR) if OR > 1 else np.sqrt(OR)
    if rr < 1:
        rr = 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))

items = [("Flat vs control", 13.96), ("Modulated vs control", 16.19), ("Two-voice vs control", 20.92),
         ("Distance (per doubling)", 0.63), ("Group size (per doubling)", 0.70)]
ev_rows = []
for lab, orv in items:
    e = e_value(orv)
    ev_rows.append([lab, f"{orv:.2f}", f"{e:.2f}",
                    "robust" if e > 3 else "fragile"])
write_table("tab_p6_evalue", ["Contrast", "Odds ratio", "E-value", "Robustness"], ev_rows,
            "E-values for the main playback effects: the minimum strength of association that an unmeasured confounder would need with both exposure and outcome to explain away the estimate.",
            "tab:p6evalue", "lccc")

# ------------------------------------------------------------------ IPW for large vs small groups
d["Large"] = (d["Npenguins"] > 8).astype(int)
ps_form = "Large ~ C(CallType) + C(Context) + log2D"
ps = smf.logit(ps_form, data=d).fit(disp=0)
d["ps"] = ps.predict(d)
p1 = d["Large"].mean()
d["w"] = d["Large"] / d["ps"] * p1 + (1 - d["Large"]) / (1 - d["ps"]) * (1 - p1)
lo, hi = np.percentile(d["w"], [1, 99])
d["w"] = d["w"].clip(lo, hi)

crude = smf.logit("ResponseYesNo ~ Large", data=d).fit(disp=0)
adj = smf.logit("ResponseYesNo ~ Large + C(CallType) + C(Context) + log2D", data=d).fit(disp=0)
wgt = smf.glm("ResponseYesNo ~ Large", data=d, family=sm.families.Binomial(),
              freq_weights=d["w"]).fit(cov_type="cluster", cov_kwds={"groups": d["expID"]})

def orci(res, term="Large"):
    b, se = res.params[term], res.bse[term]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)

# cluster bootstrap
boot = []
groups = d["expID"].unique()
for _ in range(400):
    ids = rng.choice(groups, size=len(groups), replace=True)
    bb = pd.concat([d[d.expID == g] for g in ids], ignore_index=True)
    try:
        pe = smf.logit(ps_form, data=bb).fit(disp=0)
        bb["ps"] = pe.predict(bb)
        p1b = bb["Large"].mean()
        bb["w"] = bb["Large"] / bb["ps"] * p1b + (1 - bb["Large"]) / (1 - bb["ps"]) * (1 - p1b)
        bb["w"] = bb["w"].clip(*np.percentile(bb["w"], [1, 99]))
        fit = smf.glm("ResponseYesNo ~ Large", data=bb, family=sm.families.Binomial(),
                      freq_weights=bb["w"]).fit()
        boot.append(np.exp(fit.params["Large"]))
    except Exception:
        continue
b0 = orci(wgt)[0]
blo, bhi = np.percentile(boot, [2.5, 97.5])

# control-only negative control
ctrl = d[d.Type == "control"].copy()
ctrl["log2D"] = np.log2(ctrl.Distance); ctrl["log2N"] = np.log2(ctrl.Npenguins)
neg = smf.logit("ResponseYesNo ~ log2N + log2D + C(Context)", data=ctrl).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": ctrl["expID"]})
b, se = neg.params["log2N"], neg.bse["log2N"]

ipw_rows = []
for lab, res in [("Crude", crude), ("Adjusted", adj), ("IPW (marginal)", wgt)]:
    o, l, h = orci(res)
    if lab == "IPW (marginal)":
        l, h = blo, bhi
    ipw_rows.append([lab, f"{o:.2f}", f"{l:.2f}--{h:.2f}"])
ipw_rows.append(["Control trials only", f"{np.exp(b):.2f}",
                 f"{np.exp(b-1.96*se):.2f}--{np.exp(b+1.96*se):.2f}"])
write_table("tab_p6_ipw", ["Model", "OR (large vs small group)", "95\\% CI"], ipw_rows,
            "Effect of belonging to a large group ($N>8$) on any response: crude, covariate-adjusted, inverse-propensity-weighted, and a control-only negative-control test.",
            "tab:p6ipw", "lcc")

# covariate balance / overlap diagnostics
X = patsy.dmatrix(ps_form.split("~")[1].strip(), d, return_type="dataframe").drop(columns="Intercept")


def wsmd(col, w):
    x = X[col].values.astype(float)
    t = d["Large"].values.astype(bool)
    w = np.asarray(w, dtype=float)
    mt = np.sum(w[t] * x[t]) / np.sum(w[t])
    mc = np.sum(w[~t] * x[~t]) / np.sum(w[~t])
    vt = np.sum(w[t] * (x[t] - mt) ** 2) / np.sum(w[t])
    vc = np.sum(w[~t] * (x[~t] - mc) ** 2) / np.sum(w[~t])
    den = np.sqrt((vt + vc) / 2)
    return (mt - mc) / den if den > 1e-9 else 0.0


pretty = {"C(CallType)[T.flat]": "Call type: flat", "C(CallType)[T.mod]": "Call type: modulated",
          "C(CallType)[T.tv]": "Call type: two-voice", "C(Context)[T.searching]": "Context: searching",
          "C(Context)[T.after]": "Context: after", "log2D": "log2(distance)"}
smd_rows = [[pretty.get(c, c), f"{wsmd(c, np.ones(len(d))):.2f}", f"{wsmd(c, d['w'].values):.2f}"]
            for c in X.columns]
write_table("tab_p6_smd", ["Covariate", "SMD before", "SMD after IPW"], smd_rows,
            "Standardised mean differences between large and small groups before and after inverse-propensity weighting; values below $0.10$ indicate adequate balance.",
            "tab:p6smd", "lcc")
max_before = max(abs(wsmd(c, np.ones(len(d)))) for c in X.columns)
max_after = max(abs(wsmd(c, d["w"].values)) for c in X.columns)
overlap = {g: [float(d.loc[d.Large == g, "ps"].min()), float(d.loc[d.Large == g, "ps"].max())]
           for g in [0, 1]}
print("max |SMD| before/after:", round(max_before, 3), round(max_after, 3))
print("propensity overlap:", overlap)

# ------------------------------------------------------------------ response cascade
# None -> H -> V -> A conditional transitions
mH = smf.logit("HeadsUpYesNo ~ C(CallType, Treatment('control')) + C(Context, Treatment('commuting')) + log2D + log2N",
               data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["expID"]})
h1 = d[d.HeadsUpYesNo == 1].copy()
mV = smf.logit("VocalYesNo ~ C(CallType, Treatment('control')) + C(Context, Treatment('commuting')) + log2D + log2N",
               data=h1).fit(disp=0, cov_type="cluster", cov_kwds={"groups": h1["expID"]})
v1 = d[(d.VocalYesNo == 1) & (d.Type == "call")].copy()
v1["CallType"] = v1["CallType"].cat.remove_unused_categories()
mA = smf.logit("ApproachYesNo ~ C(CallType, Treatment('flat')) + C(Context, Treatment('commuting')) + log2D + log2N",
               data=v1).fit(disp=0, cov_type="cluster", cov_kwds={"groups": v1["expID"]})


def pred(res, ct, dist, n=8):
    nd = pd.DataFrame({"CallType": [ct], "Context": ["searching"], "Distance": [dist],
                       "log2D": [np.log2(dist)], "log2N": [np.log2(n)], "Npenguins": [n]})
    return float(res.predict(nd).iloc[0])


casc_rows = []
casc_res = {}
for ct in ["control"] + CTYPES:
    ph = pred(mH, ct, 100)
    pv = pred(mV, ct, 100)
    pa = 0.0 if ct == "control" else pred(mA, ct, 100)
    casc = ph * pv * pa
    casc_res[ct] = dict(H=ph, VH=pv, AV=pa, HVA=casc)
    casc_rows.append([CALLS[ct]["label"], f"{100*ph:.1f}", f"{100*pv:.1f}",
                      f"{100*pa:.1f}", f"{100*casc:.1f}"])
write_table("tab_p6_cascade", ["Call type", "$P(H)$", "$P(V\\mid H)$", "$P(A\\mid V)$",
                               "$P(H,V,A)$"], casc_rows,
            "Conditional response probabilities at 100 m, $N=8$, searching context (percentages); the last column is the product of the three, not a time-stamped sequence.",
            "tab:p6casc", "lcccc")

# figure: transition probabilities vs distance
dd = np.linspace(20, 200, 80)
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), sharey=True)
for ax, (res, lab) in zip(axes, [(mH, "$P(H)$"), (mV, "$P(V\\mid H)$"), (mA, "$P(A\\mid V)$")]):
    for ct in CTYPES:
        vals = [pred(res, ct, x) for x in dd]
        ax.plot(dd, vals, color=CALLS[ct]["color"], label=CALLS[ct]["label"])
    ax.set_xlabel("Distance (m)"); ax.set_title(lab); ax.set_ylim(0, 1)
axes[0].set_ylabel("Transition probability"); axes[0].legend(fontsize=7)
fig.tight_layout(); save(fig, "fig_p6_cascade")

dump("p6", dict(evalue={k: e_value(v) for k, v in items},
                ipw=dict(orv=b0, lo=float(blo), hi=float(bhi), crude=float(np.exp(crude.params['Large'])),
                         neg_control=float(np.exp(b))),
                cascade=casc_res))
print(json.dumps(casc_res, indent=2))
