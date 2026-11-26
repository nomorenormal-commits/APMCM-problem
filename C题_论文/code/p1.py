"""Problem 1: acoustic description, nonparametric tests, classification."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import wavfile
from scipy.signal import spectrogram
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                     cross_val_predict, StratifiedKFold, cross_val_score)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             balanced_accuracy_score, f1_score)
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

from common import (DATA, CTYPES, CALLS, load_calls, set_style, save, write_table, dump, RES, fmt_p)

set_style()
rng = np.random.RandomState(42)

df = load_calls()
df["BW"] = df["Q75"] - df["Q25"]
df["FpeakF01"] = df["Fpeak"] / df["F01"]
df["Q50F01"] = df["Q50"] / df["F01"]
df["FpeakQ50"] = df["Fpeak"] / df["Q50"]
df["Q75Q25"] = df["Q75"] / df["Q25"]

MEAS = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75", "PulseRate", "F02"]
NICE = {"Duration": "Duration (s)", "F01": "F01 (Hz)", "Fpeak": "Peak frequency (Hz)",
        "Q25": "Q25 (Hz)", "Q50": "Q50 (Hz)", "Q75": "Q75 (Hz)",
        "PulseRate": "Pulse rate (s$^{-1}$)", "F02": "F02 (Hz)"}

# ---------------- descriptive table ----------------
rows = []
for m in MEAS:
    row = [NICE[m]]
    for ct in CTYPES:
        s = df.loc[df.CallType == ct, m].dropna()
        if len(s) == 0:
            row.append("--")
        else:
            row.append(f"{s.median():.2f} ({s.quantile(.25):.2f}--{s.quantile(.75):.2f}), {len(s)}")
    rows.append(row)
write_table("tab_p1_desc", ["Measure", "Flat", "Modulated", "Two-voice"], rows,
            "Descriptive statistics of sea-surface call measures (median, interquartile range, and sample size).",
            "tab:p1desc", "lccc", size="\\footnotesize")

# ---------------- nonparametric tests ----------------
kw_rows, pairwise_rows = [], []
eps2 = {}
for m in ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75", "BW", "FpeakF01"]:
    groups = [df.loc[df.CallType == ct, m].dropna().values for ct in CTYPES]
    H, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    e2 = max(0.0, (H - len(groups) + 1) / (n - len(groups)))
    eps2[m] = e2
    e2txt = f"{e2:.3f}" if e2 >= 0.0005 else "$<$0.001"
    kw_rows.append([NICE.get(m, m.replace("FpeakF01", "Fpeak/F01")), f"{H:.2f}", fmt_p(p), e2txt])

# pulse rate is available for modulated and two-voice only
a = df.loc[df.CallType == "mod", "PulseRate"].dropna()
b = df.loc[df.CallType == "tv", "PulseRate"].dropna()
U, p = stats.mannwhitneyu(a, b)
kw_rows.append([NICE["PulseRate"] + " (mod vs tv)", f"{U:.1f}", fmt_p(p),
                f"{(2*U)/(len(a)*len(b))-1:+.3f}"])

# pairwise Mann-Whitney with Holm for the six universally measured features
for m in ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]:
    ps, cells = [], []
    for i, j in [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]:
        a = df.loc[df.CallType == i, m].dropna()
        b = df.loc[df.CallType == j, m].dropna()
        U, p = stats.mannwhitneyu(a, b)
        r = 2 * U / (len(a) * len(b)) - 1
        ps.append(p); cells.append((i, j, p, r))
    rej, padj, _, _ = multipletests(ps, method="holm")
    for (i, j, p, r), pa, sig in zip(cells, padj, rej):
        pairwise_rows.append([NICE[m], f"{i}--{j}", fmt_p(p), fmt_p(pa),
                              f"{r:+.2f}", "yes" if sig else "no"])

write_table("tab_p1_tests", ["Measure", "$H$ or $U$", "$p$", "Effect size"], kw_rows,
            "Kruskal--Wallis tests across the three call types (first block) and Mann--Whitney tests for measures available in two types (second block). Effect size is $\\varepsilon^2$ for Kruskal--Wallis and the rank-biserial correlation for Mann--Whitney.",
            "tab:p1tests", "lrrr")

write_table("tab_p1_pairwise", ["Measure", "Comparison", "Raw $p$", "Holm $p$", "$r_{rb}$", "Significant"], pairwise_rows,
            "Pairwise Mann--Whitney comparisons with Holm correction (universally measured features).",
            "tab:p1pair", "llrrrc", place="[H]")

# ---------------- classification ----------------
base_feats = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75", "BW", "FpeakF01", "Q50F01", "FpeakQ50", "Q75Q25"]
ext_feats = base_feats + ["F02", "PulseRate", "hasF02", "hasPR"]
df["hasF02"] = df["F02"].notna().astype(int)
df["hasPR"] = df["PulseRate"].notna().astype(int)

Xb, Xe, y = df[base_feats].values, df[ext_feats].values, df["CallType"].values
imp_b = SimpleImputer(strategy="median")
imp_e = SimpleImputer(strategy="median")
Xb = imp_b.fit_transform(Xb)
Xe = imp_e.fit_transform(Xe)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
models = {
    "Logistic regression (reduced)": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                                ("sc", StandardScaler()),
                                                ("clf", LogisticRegression(max_iter=5000,
                                                                           class_weight="balanced"))]), Xb),
    "Random forest (reduced)": (RandomForestClassifier(n_estimators=600, class_weight="balanced",
                                                       random_state=42, min_samples_leaf=2), Xb),
    "Random forest (full)": (RandomForestClassifier(n_estimators=600, class_weight="balanced",
                                                    random_state=42, min_samples_leaf=2), Xe),
    "Extra trees (full)": (ExtraTreesClassifier(n_estimators=600, class_weight="balanced",
                                                random_state=42, min_samples_leaf=2), Xe),
    "Gradient boosting (full)": (GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                                            max_depth=3, random_state=42), Xe),
}
cls_rows, cls_res = [], {}
for name, (model, X) in models.items():
    sc = cross_validate(model, X, y, cv=cv,
                        scoring=["accuracy", "balanced_accuracy", "f1_macro"], n_jobs=1)
    cls_rows.append([name, f"{sc['test_accuracy'].mean():.3f} $\\pm$ {sc['test_accuracy'].std():.3f}",
                     f"{sc['test_balanced_accuracy'].mean():.3f} $\\pm$ {sc['test_balanced_accuracy'].std():.3f}",
                     f"{sc['test_f1_macro'].mean():.3f} $\\pm$ {sc['test_f1_macro'].std():.3f}"])
    cls_res[name] = dict(acc=sc["test_accuracy"].mean(), acc_sd=sc["test_accuracy"].std(),
                         bal=sc["test_balanced_accuracy"].mean(), f1=sc["test_f1_macro"].mean())

# hierarchical, deployable rule: two-voice is identified by a measurable second
# fundamental; flat and modulated are then separated by a random forest on the
# universally measured spectral features.
n_tv = int((df.CallType == "tv").sum())
mask = (df.CallType != "tv").values
rf2 = RandomForestClassifier(n_estimators=600, class_weight="balanced", random_state=42,
                             min_samples_leaf=2)
acc2 = cross_val_score(rf2, Xb[mask], y[mask], cv=RepeatedStratifiedKFold(
    n_splits=5, n_repeats=10, random_state=42), scoring="accuracy")
pred2 = cross_val_predict(rf2, Xb[mask], y[mask], cv=StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42))
yhat_h = np.array(["tv"] * len(y), dtype=object)
yhat_h[mask] = pred2
bal_h = balanced_accuracy_score(y, yhat_h)
f1_h = f1_score(y, yhat_h, average="macro")
overall = (n_tv + acc2.mean() * mask.sum()) / len(df)
cls_rows.append(["Hierarchical rule + RF", f"{overall:.3f} $\\pm$ {acc2.std()*mask.sum()/len(df):.3f}",
                 f"{bal_h:.3f}", f"{f1_h:.3f}"])
cls_res["Hierarchical rule + RF"] = dict(acc=overall, acc_sd=acc2.std() * mask.sum() / len(df),
                                         stage2=acc2.mean())

write_table("tab_p1_classifier", ["Model", "Accuracy", "Balanced accuracy", "Macro $F_1$"], cls_rows,
            "Repeated stratified 5-fold cross-validation (10 repeats) for three-class call-type classification.",
            "tab:p1clf", "lccc", size="\\footnotesize",
            notes="Mean $\\pm$ standard deviation over 50 folds. ``Reduced'' uses only universally measured continuous features; ``full'' adds F02, pulse rate and indicators of whether these structural cues are measurable; the hierarchical rule assigns two-voice from the presence of F02 and then separates the other two types with a random forest.")

# confusion matrix and importance for the full random forest
rf = RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=42, min_samples_leaf=2)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
yhat = cross_val_predict(rf, Xe, y, cv=cv5)
cm = confusion_matrix(y, yhat, labels=CTYPES)
rf.fit(Xe, y)
perm = permutation_importance(rf, Xe, y, n_repeats=40, random_state=42, scoring="balanced_accuracy")
order = np.argsort(perm.importances_mean)[::-1]

rf_red = RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=42,
                                min_samples_leaf=2).fit(Xb, y)
perm_red = permutation_importance(rf_red, Xb, y, n_repeats=40, random_state=42,
                                  scoring="balanced_accuracy")
order_red = np.argsort(perm_red.importances_mean)[::-1]

nice_imp = {"F02": "$F_{02}$", "PulseRate": "Pulse rate", "hasF02": "second fundamental present",
            "hasPR": "pulse train present", "FpeakF01": "$F_{peak}/F_{01}$", "Q50F01": "$Q_{50}/F_{01}$",
            "FpeakQ50": "$F_{peak}/Q_{50}$", "Q75Q25": "$Q_{75}/Q_{25}$", "Fpeak": "Peak frequency",
            "BW": "Bandwidth", "F01": "$F_{01}$", "Q25": "$Q_{25}$", "Q50": "$Q_{50}$",
            "Q75": "$Q_{75}$", "Duration": "Duration"}
imp_rows = []
for i in order[:6]:
    imp_rows.append(["Full", nice_imp.get(ext_feats[i], ext_feats[i]),
                     f"{perm.importances_mean[i]:.3f}"])
for i in order_red[:6]:
    imp_rows.append(["Reduced", nice_imp.get(base_feats[i], base_feats[i]),
                     f"{perm_red.importances_mean[i]:.3f}"])
write_table("tab_p1_importance", ["Model", "Feature", "Permutation importance"], imp_rows,
            "Leading permutation importances for the full and reduced random-forest classifiers (decrease in balanced accuracy when a feature is permuted).",
            "tab:p1imp", "llr", place="[H]")

# PCA on reduced features
imp = SimpleImputer(strategy="median")
Xp = StandardScaler().fit_transform(imp.fit_transform(Xb))
pca = PCA(n_components=2).fit(Xp)
Z = pca.transform(Xp)

# ---------------- figures ----------------
# (1) real spectrograms
files = {
    "flat": "Supplementary5_Audio_recordings_of_African_penguins’_sea-surfacecalls_flat.wav",
    "mod": "Supplementary5_Audio_recordings_of_African_penguins’_sea-surfacecalls_modulated.wav",
    "tv": "Supplementary5_Audio_recordings_of_African_penguins’_sea-surfacecalls_two-voice.wav",
}
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15), sharey=True)
for ax, ct in zip(axes, CTYPES):
    sr, x = wavfile.read(DATA / files[ct])
    x = x.astype(float)
    f, t, S = spectrogram(x, fs=sr, nperseg=256, noverlap=200, window="hann")
    S = 10 * np.log10(S + 1e-10)
    im = ax.pcolormesh(t * 1000, f, S, shading="gouraud", cmap="magma",
                       vmin=np.percentile(S, 5), vmax=np.percentile(S, 99.5))
    ax.set_title(CALLS[ct]["label"], color=CALLS[ct]["color"])
    ax.set_xlabel("Time (ms)")
    ax.set_ylim(0, 4000)
    if ct == "tv":
        ax.annotate("two fundamentals\n275/304 Hz ($\\Delta$=29 Hz)",
                    xy=(520, 290), xytext=(470, 1450), ha="center", fontsize=5.2,
                    color="white", arrowprops=dict(arrowstyle="->", color="white", lw=0.6))
axes[0].set_ylabel("Frequency (Hz)")
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
cbar.set_label("Power (dB)")
save(fig, "fig_p1_spectrograms")

# (2) distributions
panels = [("Duration", "Duration (s)"), ("F01", "$F_{01}$ (Hz)"), ("Fpeak", "Peak frequency (Hz)"),
          ("Q50", "$Q_{50}$ (Hz)"), ("BW", "Bandwidth $Q_{75}-Q_{25}$ (Hz)"),
          ("FpeakF01", "$F_{peak}/F_{01}$")]
fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.1))
for ax, (m, lab) in zip(axes.ravel(), panels):
    data = [df.loc[df.CallType == ct, m].dropna().values for ct in CTYPES]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for pc, ct in zip(parts["bodies"], CTYPES):
        pc.set_facecolor(CALLS[ct]["color"]); pc.set_alpha(0.28); pc.set_edgecolor("none")
    bp = ax.boxplot(data, widths=0.16, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1))
    for patch, ct in zip(bp["boxes"], CTYPES):
        patch.set_facecolor("white"); patch.set_edgecolor(CALLS[ct]["color"])
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels([CALLS[c]["label"] for c in CTYPES])
    ax.set_ylabel(lab)
fig.tight_layout()
save(fig, "fig_p1_dist")

# (3) PCA + confusion + importance
fig = plt.figure(figsize=(7.1, 2.55))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.9, 1.5], wspace=0.45)
ax = fig.add_subplot(gs[0])
for ct in CTYPES:
    m = y == ct
    ax.scatter(Z[m, 0], Z[m, 1], s=16, alpha=0.75, color=CALLS[ct]["color"],
               marker=CALLS[ct]["marker"], label=CALLS[ct]["label"], edgecolor="none")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
ax.legend(loc="best", handletextpad=0.2)
ax = fig.add_subplot(gs[1])
ConfusionMatrixDisplay(cm, display_labels=["Flat", "Mod.", "2-voice"]).plot(
    ax=ax, colorbar=False, cmap="Blues", values_format="d")
ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
ax.set_title("5-fold predictions", fontsize=9)
ax = fig.add_subplot(gs[2])
top = order[:7][::-1]
nice = {"hasF02": "F02 present", "hasPR": "pulse train present", "FpeakF01": "$F_{peak}/F_{01}$",
        "Q50F01": "$Q_{50}/F_{01}$", "FpeakQ50": "$F_{peak}/Q_{50}$", "Q75Q25": "$Q_{75}/Q_{25}$",
        "Fpeak": "Peak frequency", "BW": "Bandwidth", "F01": "$F_{01}$", "Q25": "$Q_{25}$",
        "Q50": "$Q_{50}$", "Q75": "$Q_{75}$", "Duration": "Duration"}
ax.barh([nice.get(ext_feats[i], ext_feats[i]) for i in top],
        perm.importances_mean[top], xerr=perm.importances_std[top],
        color="#6b8fb5", edgecolor="none", height=0.65, error_kw=dict(lw=0.7))
for y, i in enumerate(top):
    v = perm.importances_mean[i]
    ax.text(v + perm.importances_std[i] + 0.008, y, f"{v:.3f}", va="center", fontsize=6)
ax.set_xlim(0, max(perm.importances_mean[top] + perm.importances_std[top]) * 1.45)
ax.tick_params(axis="y", labelsize=7)
ax.set_xlabel("Permutation importance")
fig.tight_layout()
save(fig, "fig_p1_classifier")

# ---------------- write results ----------------
summary = {
    "n": len(df),
    "kw": {m: (eps2.get(m, np.nan)) for m in eps2},
    "classifier": cls_res,
    "confusion": cm.tolist(),
    "importance": {f: float(v) for f, v in zip(ext_feats, perm.importances_mean)},
    "importance_reduced": {f: float(v) for f, v in zip(base_feats, perm_red.importances_mean)},
    "pca_var": pca.explained_variance_ratio_.tolist(),
    "stage2_acc": float(acc2.mean()),
}
dump("p1", summary)
print(open(RES / "p1.json").read())
