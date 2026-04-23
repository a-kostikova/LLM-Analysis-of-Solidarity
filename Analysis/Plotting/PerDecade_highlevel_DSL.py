from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
import json

"""
Plot decade-level DSL trends from saved outputs.

Required inputs
---------------
INPUT_JSON
    Full metadata file used to recover document-level information such as `id` and `year`.
    The saved DSL arrays do not contain year metadata, so this file is needed to assign
    each retained document to a decade.

YTILDE_FILE
    NumPy array containing the DSL-adjusted class scores for all retained samples.
    These are the values aggregated by decade for plotting.

IDS_FILE
    NumPy array of document IDs aligned row-wise with `YTILDE_FILE`.
    This file is used to merge the DSL scores back to the metadata in `INPUT_JSON`.

Why all three files are needed
------------------------------
The DSL output array stores adjusted class scores, but not the original metadata.
The ID file provides the row-to-document mapping, and the full JSON provides the
document metadata needed for temporal aggregation. Together, these files allow us
to reconstruct decade-level trends from the saved DSL outputs.
"""

# Config
INPUT_JSON = "Migrant_1867-2025_AllModels.json"
YTILDE_FILE = "dsl_outputs/Ytilde_full_high.npy"
IDS_FILE = "dsl_outputs/ids_full_high.npy"

TITLE = "Migrant Solidarity Over Time (Soft-Label DSL)"

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

labels = ["solidarity", "anti-solidarity", "mixed", "none"]

df_dsl = pd.DataFrame(Ytilde, columns=labels)
df_dsl["id"] = ids_kept

# Merge years onto DSL rows
df = df_dsl.merge(meta, on="id", how="left")

# Prepare decade-level table
df["decade"] = (df["year"] // 10) * 10
dft = df.groupby("decade")[labels].mean().sort_index()

# Ensure full decade axis
min_decade = int((dft.index.min() // 10) * 10)
max_decade = int((dft.index.max() // 10) * 10)
full_decades = list(range(min_decade, max_decade + 1, 10))
dft = dft.reindex(full_decades)

# Plot
series_styles = {
    'solidarity':      {'label': 'Solidarity',      'color': '#267f53', 'marker': 'o'},
    'anti-solidarity': {'label': 'Anti-solidarity', 'color': '#f35695', 'marker': '^'},
    'mixed':           {'label': 'Mixed',           'color': '#ff9b28', 'marker': '+'},
    # 'none':          {'label': 'None',            'color': '#9aa0a6', 'marker': 's'},
}

for key in series_styles:
    if key not in dft.columns:
        dft[key] = 0.0

line_width = 5
marker_size = 10

plt.figure(figsize=(7, 4.5))
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.82)
ax = plt.subplot(1, 1, 1)
ax.tick_params(axis='y', labelsize=12)

x = list(range(len(full_decades)))
decade_to_x = {dec: i for i, dec in enumerate(full_decades)}

if 1920 in decade_to_x and 1940 in decade_to_x:
    left_edge = decade_to_x[1920]
    right_edge = decade_to_x[1940]
    if right_edge > left_edge:
        ax.axvspan(left_edge, right_edge, color='lightgrey', alpha=0.3, zorder=0)

pre_gap_decades = [d for d in full_decades if d <= 1920]
post_gap_decades = [d for d in full_decades if d >= 1940]

def plot_segment(decades_subset, add_legend=False):
    xs = [decade_to_x[d] for d in decades_subset]
    for key, st in series_styles.items():
        ys = dft.loc[decades_subset, key].values
        ax.plot(
            xs, ys,
            linestyle='-', linewidth=line_width,
            color=st['color'],
            marker=st['marker'], markersize=marker_size,
            markerfacecolor='white', markeredgewidth=2, markeredgecolor=st['color'],
            label=(st['label'] if add_legend else None)
        )

plot_segment(pre_gap_decades, add_legend=True)
plot_segment(post_gap_decades, add_legend=False)

ax.set_xlim(0, len(full_decades) - 1)
ax.set_xticks(x)
ax.set_xticklabels([f"{d}s" for d in full_decades], rotation=45)
ax.set_ylim((0, 1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

plt.xlabel("Decade")
plt.ylabel("Percentage")
plt.title(TITLE, fontsize=14)
plt.legend(loc="upper left", fontsize="medium")
plt.tight_layout()
plt.savefig("SoftLabelDSL_Fullscale_Highlevel.pdf", bbox_inches="tight")
plt.show()