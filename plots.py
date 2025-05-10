from tueplots import bundles, figsizes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns     # only for colour‑mapping convenience
import os
plt.rcParams.update(bundles.neurips2023(usetex=False))

def bars_combined_only(df,
                       scenario,
                       metrics=("Macro F1", "Micro F1", "Exact Match")):
    """
    Single grouped‑bar chart that shows the *Combined* sample‑weighted
    scores for one scenario.  Expects that df already contains rows with
    Dataset == 'Combined' (see the weighting code we added earlier).
    """
    sub = (
        df[(df["Dataset"] == "Combined") & (df["Scenario"] == scenario)]
          .pivot(index="Model", columns="Metric", values="Value")
          .sort_index()
    )

    # Keep only the requested metrics that exist
    cols = [m for m in metrics if m in sub.columns]
    if not cols:
        raise ValueError(f"No matching metrics {metrics} for scenario '{scenario}'")

    ax = sub[cols].plot(
        kind="bar",
        width=0.8,
        edgecolor="black",
    )
    ax.set_title(f"{scenario}")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=len(cols))
    plt.tight_layout()
    plt.show()

def radar_combined_fixed(df, scenario,
                         metrics=("Macro F1", "Micro F1", "Exact Match")):
    """
    Combined dataset, one polygon per model, fixed 0‑to‑1 radius.
    """
    from math import pi
    import matplotlib.pyplot as plt

    sub = df[
        (df["Dataset"] == "Combined") &
        (df["Scenario"] == scenario) &
        (df["Metric"].isin(metrics))
    ]

    metrics = list(metrics)                    # preserve order
    models  = sub["Model"].unique().tolist()
    table = (
        sub.pivot(index="Model", columns="Metric", values="Value")
           .reindex(index=models, columns=metrics)
           .fillna(0)
           .values
    )

    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    for row, model in zip(table, models):
        values = row.tolist() + [row[0]]
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.25)

    # ----- fixed 0‑1 scale & nicer grid -----
    ax.set_ylim(0, 1)                     # lock radius
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])   # grid rings
    ax.set_yticklabels([".2", ".4", ".6", ".8"])
    ax.set_rlabel_position(225)           # move radial labels away from first axis

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title(f"{scenario}", y=1.1)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

def radar_metric_by_scenario(df, metric="Macro F1", dataset="Combined"):
    """
    For the chosen `metric`, plot one radar:
    * axes  ........ scenarios where the metric exists
    * polygons ..... models
    * radius ........ metric value (0‑1, fixed)
    """
    from math import pi
    import matplotlib.pyplot as plt

    sub = df[
        (df["Dataset"] == dataset) &
        (df["Metric"] == metric)
    ]

    scenarios = sub["Scenario"].unique().tolist()
    scenarios.sort()                               # consistent order
    models = sub["Model"].unique().tolist()

    table = (
        sub.pivot(index="Model", columns="Scenario", values="Value")
           .reindex(index=models, columns=scenarios)
           .fillna(0)
           .values
    )

    N = len(scenarios)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    for row, model in zip(table, models):
        values = row.tolist() + [row[0]]
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.25)

    # fixed scale 0‑1
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([ ".2", ".4", ".6", ".8"])
    ax.set_rlabel_position(225)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_title(f"{metric}", y=1.08)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


# ---------- example call ----------
base_path = "/Users/fagunpatel/Desktop"
df_long = pd.read_csv(os.path.join(base_path, "slp_eval_results_long.csv"))
df_long["Metric"] = (
    df_long["Metric"]
    .str.strip()          # remove leading/trailing spaces
    .str.replace("_", " ")
    .str.title()          # 'macro f1' -> 'Macro F1'
)

sample_sizes = {
    "UltraSuite": 1000,
    "ENNI":       1000,
    "PERCEPT-GFTA":   1000,   # change the key to your real dataset name
    "LeNormand (French)":   400,
}

# --- 2) attach weights ---
df_long["Weight"] = df_long["Dataset"].map(sample_sizes)

# (optional) check nothing is missing
assert df_long["Weight"].isna().sum() == 0, "Add sample size for every dataset!"

# --- 3) weighted average per Scenario–Model–Metric ---
agg = (
    df_long
      .groupby(["Scenario", "Model", "Metric"], as_index=False)
      .apply(lambda g: (g["Value"] * g["Weight"]).sum() / g["Weight"].sum())
      .rename(columns={None: "Value"})        # result column from apply
)

# --- 4) label as one pseudo‑dataset so you can plot it too ---
agg["Dataset"] = "Combined"

# --- 5) merge back if you want the combined scores alongside originals ---
df_with_combined = pd.concat([df_long, agg], ignore_index=True)

scenarios = ["Binary Classification", "ASR-Based Classification", "Disorder Type Classification", "Disorder Symptom Classification"]

# for scenario in scenarios:
#     # bars_combined_only(df_with_combined, scenario)
#     radar_combined_fixed(df_with_combined, scenario)

metrics = ["Macro F1", "Micro F1", "Exact Match"]

for metric in metrics:
    radar_metric_by_scenario(df_with_combined, metric)

