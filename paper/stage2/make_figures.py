"""Build paper figures from stats.json (output of stage2_stats.py).

  python make_figures.py [--stats out/stats.json] [--ablation out/ablation_retention.json] [--out figures]

Paths are relative to the current directory.
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--stats", default="out/stats.json")
ap.add_argument("--ablation", default="out/ablation_retention.json")
ap.add_argument("--out", default="figures")
args = ap.parse_args()
S = json.loads(Path(args.stats).read_text())["systems"]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})

# Figure: per-field accuracy on held-out 256
fields = ["ticket_id", "customer_tier", "category", "severity", "order_ref", "refund_amount_usd", "language"]
labels = ["id", "tier", "category", "severity", "order", "refund", "lang"]
v2dpo = {f: sum(S[f"v2-s{i}/dpo"]["per_field_accuracy"][f] for i in (1, 2, 3)) / 3 for f in fields}
series = [("Base", S["v1/base"]["per_field_accuracy"], "#bbbbbb"),
          ("v1 DPO", S["v1/dpo"]["per_field_accuracy"], "#6a8fc7"),
          ("v2 DPO (3-seed mean)", v2dpo, "#1f3f77")]
fig, ax = plt.subplots(figsize=(3.1, 1.9))
w = 0.27
for j, (name, d, c) in enumerate(series):
    ax.bar([i + (j - 1) * w for i in range(len(fields))], [100 * d[f] for f in fields], w, label=name, color=c)
ax.set_xticks(range(len(fields)))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylim(30, 104)
ax.set_ylabel("Accuracy (%)")
ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.2)
fig.savefig(out / "per_field.pdf")

# Figure: held-out field match vs blind injections refused, per system
fig, ax = plt.subplots(figsize=(3.1, 2.2))
pts = [("v1/sft", "o", "#6a8fc7", None), ("v1/dpo", "s", "#6a8fc7", ("v1 SFT, v1 DPO", (-4, 6)))]
lab = {1: {"sft": ("v2 s1, s2 SFT", (-30, 6)), "dpo": ("v2 s1 DPO", (-14, 6))},
       2: {"sft": None, "dpo": ("v2 s2 DPO", (4, -9))},
       3: {"sft": ("v2 s3 SFT", (-40, 3)), "dpo": ("v2 s3 DPO", (4, -9))}}
for i in (1, 2, 3):
    pts += [(f"v2-s{i}/sft", "o", "#c0504d", lab[i]["sft"]), (f"v2-s{i}/dpo", "s", "#1f3f77", lab[i]["dpo"])]
for key, m, c, l in pts:
    s_ = S[key]
    x = 100 * s_["field_exact_match"]
    lo, hi = (100 * v for v in s_["fem_ci95"])
    y = s_["hostile_v2"]["injections_refused"]
    ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt=m, color=c, ms=3.5, capsize=1.5, lw=0.7)
    if l:
        ax.annotate(l[0], (x, y), textcoords="offset points", xytext=l[1], fontsize=5.5)
abl_path = Path(args.ablation)
if abl_path.exists():
    A = json.loads(abl_path.read_text())["ablation"]
    for i in (1, 2, 3):
        for m, mk in (("sft", "^"), ("dpo", "v")):
            s_ = A[f"abl-s{i}/{m}"]
            x = 100 * s_["field_exact_match"]
            lo, hi = (100 * v for v in s_["fem_ci95"])
            y = s_["hostile_v2"]["injections_refused"] + (0.12 if m == "sft" else -0.12)
            ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt=mk, color="#999999", ms=3.2, capsize=1.5, lw=0.6)
    ax.annotate("ablation s1-s3 (no adv. rows), all at 6", (95.1, 4.6), fontsize=5.5)
ax.set_xlabel("Held-out field match (%), 95% bootstrap CI")
ax.set_ylabel("Blind injections refused (/10)")
ax.set_ylim(4, 11)
ax.set_xlim(95.0, 100)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.2)
fig.savefig(out / "tradeoff.pdf")
print("ok")
