import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# 1.  Data definitions
# --------------------------
scenario1 = {
    "gpt-4o-audio":            [0.42, 0.83, 0.39],
    "gemini-2.0-flash":        [0.98, 0.98, 0.59],
    "gemini-2.0-flash-lite":   [0.89, 0.89, 0.49],
    "gpt-4o-mini-audio":       [0.43, 0.80, 0.74],
    "gpt-4o-mini-transcribe":  [0.48, 0.55, 0.20],
    "gpt-4o-transcribe":       [0.63, 0.65, 0.27]
}

scenario2 = {
    "gpt-4o-audio":            [0.0125, 0.02, 0.02],
    "gemini-2.0-flash":        [0, 0, 0],
    "gpt-4o-mini-audio":       [0.03, 0.07, 0.07],
    "gemini-2.0-flash-lite":   [0.006, 0.01, 0.01],
    "gpt-4o-transcribe":       [0.63, 0.65, 0.27],
    "Whispr":                  [0, 0, 0]
}

scenario4 = {
    "gpt-4o-audio":            [0.68, 0.75, 0.51],
    "gemini-2.0-flash":        [0.84, 0.82, 0.55],
    "gpt-4o-mini-audio":       [0.67, 0.76, 0.57],
    "gpt-4o-mini-transcribe":  [0.37, 0.44, 0.30],
    "gpt-4o-transcribe":       [0.42, 0.52, 0.36],
    "gemini-2.0-flash-lite":   [0.79, 0.77, 0.43],
    "whispr+gpt-4o":           [0.46, 0.52, 0.35]
}

scenario5 = {
    "gpt-4o-audio":            [0.57, 0.84, 0.15],
    "gemini-2.0-flash":        [0.72, 0.87, 0.11],
    "gpt-4o-mini-audio":       [0.58, 0.77, 0.64],
    "gpt-4o-mini-transcribe":  [0.23, 0.48, 0.03],
    "gpt-4o-transcribe":       [0.28, 0.59, 0.15],
    "gemini-2.0-flash-lite":   [0.71, 0.73, 0.31],
    "whispr+gpt-4o":           [0.46, 0.64, 0.17]
}

scenarios = [
    ("Scenario 1: Binary Classification", scenario1),
    ("Scenario 2: ASR‑Based Classification", scenario2),
    ("Scenario 4: Disorder Type Classification", scenario4),
    ("Scenario 5: Disorder Symptom Classification", scenario5)
]

# --------------------------
# 2.  Helper – grouped bars
# --------------------------
def extract_macro(v):
    """Return v itself if it's a float, otherwise v[0] (Macro F1)."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return v[0]
    return v

# ----------------------------------------------------------------------
# 3.  Plot – 2×2 grid, horizontal bars, no gaps
# ----------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=2, ncols=2, figsize=(11, 8), constrained_layout=True
)
axes = axes.flatten()

for idx, (ax, (title, data)) in enumerate(zip(axes, scenarios)):
    # Extract & sort by Macro F1 descending
    items        = [(model, extract_macro(val)) for model, val in data.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    models, vals = zip(*items)

    y = np.arange(len(models))
    ax.barh(y, vals, height=0.85, color="#1f77b4")

    ax.set_title(title, fontsize=10, pad=3)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()                    # highest score on top
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # x‑axis label only on bottom row (idx 2 and 3)
    if idx // 2 == 1:
        ax.set_xlabel("Macro F1", fontsize=9)

# Optional: save to file
# fig.savefig("ultrasuite_macroF1_bars_compact.png", dpi=300, bbox_inches="tight")

plt.show()

# --------------------------
# 4.  Sub‑figure B – spider plot (Macro F1 only)
# --------------------------
# Collect macro‑F1s
macro = {
    "Binary Classification": {k: v[0] for k, v in scenario1.items()},
    "ASR‑Based Classification": {k: v[0] for k, v in scenario2.items()},
    "Disorder Type Classification": {k: v[0] for k, v in scenario4.items()},
    "Disorder Symptom Classification": {k: v[0] for k, v in scenario5.items()}
}

labels    = list(macro.keys())                      # the 4 axes
models    = ["gpt-4o-audio", "gemini-2.0-flash", "gemini-2.0-flash-lite",
             "gpt-4o-mini-audio", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"]

N         = len(labels)
angles    = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles   += angles[:1]                              # close polygon

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)

for model in models:
    vals = [macro[label].get(model, 0) for label in labels]
    vals += vals[:1]
    ax.plot(angles, vals, label=model)
    ax.fill(angles, vals, alpha=0.08)

ax.set_title("UltraSuite: Macro F1 Across Scenarios", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.15))
plt.tight_layout()
plt.show()
