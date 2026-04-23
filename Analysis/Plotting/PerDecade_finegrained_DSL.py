"""
Plot decade-level fine-grained DSL trends from saved outputs.

Required inputs
---------------
INPUT_JSON
    Full metadata file used to recover document-level information such as `id` and `year`.
    The saved DSL arrays do not contain year metadata, so this file is needed to assign
    each retained document to a decade.

YTILDE_FILE
    NumPy array containing the DSL-adjusted class scores for all retained samples.

IDS_FILE
    NumPy array of document IDs aligned row-wise with `YTILDE_FILE`.

Why all three files are needed
------------------------------
The DSL output array stores adjusted class scores, but not the original metadata.
The ID file provides the row-to-document mapping, and the full JSON provides the
document metadata needed for temporal aggregation. Together, these files allow us
to reconstruct decade-level trends from the saved DSL outputs.
"""

from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
import json

# Config
INPUT_JSON = "GitHubFiles/CL/Data/ModelPredictedData/Migrant_1867-2025_AllModels.json"
YTILDE_FILE = "GitHubFiles/CL/Analysis/dsl_outputs/Ytilde_full_fine.npy"
IDS_FILE = "GitHubFiles/CL/Analysis/dsl_outputs/ids_full_fine.npy"

TITLE = "Fine-grained Solidarity Trends (Soft-Label DSL)"
NORMALIZE_WITHIN_SOLIDARITY_ANTISOLIDARITY = True

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})

# Load metadata from JSON
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

meta = pd.DataFrame([{"id": str(row["id"]), "year": int(row["year"])} for row in data])

# Load DSL outputs
Ytilde = np.load(YTILDE_FILE)
ids_kept = np.load(IDS_FILE, allow_pickle=True).astype(str)

labels = [
    's.group-based', 's.exchange-based', 's.compassionate', 's.empathic',
    'as.group-based', 'as.exchange-based', 'as.compassionate', 'as.empathic',
    'mixed.none', 'none.none'
]

df_dsl = pd.DataFrame(Ytilde, columns=labels)
df_dsl["id"] = ids_kept

# Merge years onto DSL rows
df = df_dsl.merge(meta, on="id", how="left")

# Prepare decade-level table
df["decade"] = (df["year"] // 10) * 10
dft = df.groupby("decade")[labels].mean().sort_index()

# Clip after aggregation for display
dft = dft.clip(lower=0, upper=1)

# Full decade axis
min_decade = int((dft.index.min() // 10) * 10)
max_decade = int((dft.index.max() // 10) * 10)
full_decades = list(range(min_decade, max_decade + 1, 10))
dft = dft.reindex(full_decades)

# Category splits
solidarity_labels = [c for c in dft.columns if c.startswith("s.")]
antisolidarity_labels = [c for c in dft.columns if c.startswith("as.")]

# Optional normalization within solidarity / anti-solidarity subtypes
if NORMALIZE_WITHIN_SOLIDARITY_ANTISOLIDARITY:
    cols_keep = solidarity_labels + antisolidarity_labels
    row_sums = dft[cols_keep].sum(axis=1).replace(0, np.nan)
    dft[cols_keep] = dft[cols_keep].div(row_sums, axis=0)

label_details = {
    'group-based':    {'color': '#007BA7', 'line_style': '-'},
    'exchange-based': {'color': '#D31D0D', 'line_style': '--'},
    'compassionate':  {'color': '#4CAF50', 'line_style': '-.'},
    'empathic':       {'color': '#F9AA00', 'line_style': (0, [5, 5])}
}

x_positions = list(range(len(full_decades)))
decade_to_x = {dec: i for i, dec in enumerate(full_decades)}

def plot_data(ax, label_list, title, add_legend=False):
    pre_gap_decades = [d for d in full_decades if d <= 1920]
    post_gap_decades = [d for d in full_decades if d >= 1940]

    for label in label_list:
        subtype = label.split('.')[1]
        if subtype not in label_details:
            continue
        details = label_details[subtype]

        # Pre-gap segment
        pre_idx = [d for d in pre_gap_decades if d in dft.index]
        if pre_idx:
            xs = [decade_to_x[d] for d in pre_idx]
            ys = dft.loc[pre_idx, label].values
            ax.plot(
                xs, ys,
                color=details['color'],
                linestyle=details['line_style'],
                linewidth=4,
                label=(subtype if add_legend else None)
            )

        # Post-gap segment
        post_idx = [d for d in post_gap_decades if d in dft.index]
        if post_idx:
            xs = [decade_to_x[d] for d in post_idx]
            ys = dft.loc[post_idx, label].values
            ax.plot(
                xs, ys,
                color=details['color'],
                linestyle=details['line_style'],
                linewidth=4
            )

    # Historical gap shading
    if 1920 in decade_to_x and 1940 in decade_to_x:
        left_edge = decade_to_x[1920]
        right_edge = decade_to_x[1940]
        if right_edge > left_edge:
            ax.axvspan(left_edge, right_edge, color='lightgrey', alpha=0.3, zorder=0)

    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(full_decades) - 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{d}s" for d in full_decades], rotation=45)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_xlabel("Decade")

    if add_legend:
        ax.legend(loc="upper left", fontsize="medium")

plt.figure(figsize=(9.5, 4.5))
plt.subplots_adjust(left=0.11, right=0.99, bottom=0.22, top=0.82)

ax1 = plt.subplot(1, 2, 1)
plot_data(ax1, solidarity_labels, "Solidarity", add_legend=True)

ax2 = plt.subplot(1, 2, 2)
plot_data(ax2, antisolidarity_labels, "Anti-solidarity")
ax2.set_ylabel("")

plt.suptitle(TITLE, fontsize=14)

output_path = (
    "SoftLabelDSL_Fullscale_Finegrained.pdf"
)

plt.tight_layout()
plt.savefig(output_path, bbox_inches="tight")
plt.show()