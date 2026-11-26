"""Shared utilities: paths, data loading, plotting style, model helpers."""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
TAB = ROOT / "tables"
RES = ROOT / "results"
for _d in (FIG, TAB, RES):
    _d.mkdir(exist_ok=True)

DIST = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)

CALLS = {
    "flat": dict(label="Flat", color="#1f6fb2", marker="o"),
    "mod": dict(label="Modulated", color="#d17a22", marker="s"),
    "tv": dict(label="Two-voice", color="#2e7d5b", marker="^"),
    "control": dict(label="Control", color="#7a7a7a", marker="D"),
}
CTYPES = ["flat", "mod", "tv"]
MEASURES = {
    "spec": dict(label="Spectrogram", color="#1f6fb2", marker="o",
                 file={"flat": "Data03_correlation1m_flat_call_spectrogram.csv",
                       "mod": "Data04_correlation1m_modulated_call_spectrogram.csv",
                       "tv": "Data05_correlation1m_twovoice_call_spectrogram.csv"}),
    "amp": dict(label="Amplitude spectrum", color="#d17a22", marker="s",
                file={"flat": "Data06_correlation1m_flat_call_amplitude_spectrum.csv",
                      "mod": "Data07_correlation1m_modulated_call_amplitude_spectrum.csv",
                      "tv": "Data08_correlation1m_twovoice_call_amplitude_spectrum.csv"}),
    "env": dict(label="Envelope", color="#2e7d5b", marker="^",
                file={"flat": "Data09_correlation1m_flat_call_envelope.csv",
                      "mod": "Data10_correlation1m_modulated_call_envelope.csv",
                      "tv": "Data11_correlation1m_twovoice_call_envelope.csv"}),
}


def set_style():
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
        "legend.frameon": False,
    })


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def load_calls():
    return pd.read_csv(DATA / "Data01_AfricanPenguins_seasurface_calls_acoustics.csv")


def load_playback():
    d = pd.read_csv(DATA / "Data02_AfricanPenguins_playbacks_data.csv")
    combo = []
    for _, r in d.iterrows():
        s = "".join(c for c, v in zip("HVA", [r.HeadsUpYesNo, r.VocalYesNo, r.ApproachYesNo]) if v == 1)
        combo.append(s if s else "none")
    d["Combo"] = combo
    d["CallType"] = pd.Categorical(d["CallType"], categories=["control", "flat", "mod", "tv"], ordered=True)
    d["Context"] = pd.Categorical(d["Context"], categories=["commuting", "searching", "after"], ordered=True)
    return d


def load_degradation():
    """Return long dataframe: calltype, measure, distance, corr."""
    rows = []
    for m, info in MEASURES.items():
        for ct, fn in info["file"].items():
            df = pd.read_csv(DATA / fn)
            if df.shape[1] != len(DIST):
                raise ValueError(fn)
            for j, dist in enumerate(DIST):
                v = df.iloc[:, j].dropna().values
                rows.append(pd.DataFrame({"calltype": ct, "measure": m, "distance": dist, "corr": v}))
    return pd.concat(rows, ignore_index=True)


def logistic_decay(d, f0, fmin, dh, k):
    """Hill-type decay from f0 at d=0 to fmin at infinity, half-decay at dh."""
    d = np.asarray(d, dtype=float)
    return fmin + (f0 - fmin) / (1.0 + (d / dh) ** k)


def fit_decay(d, y, p0=None):
    from scipy.optimize import curve_fit
    if p0 is None:
        p0 = [max(y[0], 0.5), 0.05, 60.0, 2.0]
    popt, pcov = curve_fit(logistic_decay, d, y, p0=p0,
                           bounds=([0, 0, 0.5, 0.1], [1.05, 0.6, 5000, 12]),
                           maxfev=100000)
    return popt, pcov


def decay_range(popt, threshold):
    f0, fmin, dh, k = popt
    if not (fmin < threshold < f0):
        return np.nan
    return dh * ((f0 - threshold) / (threshold - fmin)) ** (1.0 / k)


def r2(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)


def aicc(y, yhat, p):
    n = len(y)
    rss = np.sum((np.asarray(y) - np.asarray(yhat)) ** 2)
    rss = max(rss, 1e-12)
    aic = n * np.log(rss / n) + 2 * p
    return aic + (2 * p * (p + 1)) / max(n - p - 1, 1)


def fmt_p(p):
    if p < 1e-4:
        return "$<10^{-4}$"
    if p < 0.01:
        return f"{p:.4f}"
    return f"{p:.3f}"


def write_table(name, header, rows, caption, label, colspec=None, notes=None,
                size="\\small", place="[!ht]"):
    """Write a LaTeX table float. rows is a list of lists of str."""
    if colspec is None:
        colspec = "l" + "r" * (len(header) - 1)
    lines = [f"\\begin{{table}}{place}", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for row in rows:
        if isinstance(row, str):
            lines.append(row)
        else:
            lines.append(" & ".join(str(x) for x in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines += [f"\\par\\vspace{{2pt}}\\footnotesize {notes}"]
    lines += ["\\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def dump(name, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=2, default=conv), encoding="utf-8")
