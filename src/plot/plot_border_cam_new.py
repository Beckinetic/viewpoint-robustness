import pickle
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from matplotlib.ticker import MaxNLocator

# Set seaborn style
sns.set_theme(style="whitegrid")
plt.rcParams['ytick.left'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = '500'  # medium weight
plt.rcParams['font.stretch'] = 'semi-expanded'  # slightly expanded
plt.rcParams['figure.dpi'] = 300

order = ['Fixed', 'Extra R.', 'Restricted', 'Full']

data_path = f'../../plots/border_cam_plot_data.pkl'
plot_data = pd.read_pickle(data_path)
df_matched = plot_data['df_matched']
df_ood = plot_data['df_ood']

df_matched_ec = df_matched[df_matched['CAM Type'] == 'ec']
df_matched_ec['Data'] = 'ID'
df_matched_ic_bc = df_matched[df_matched['CAM Type'] != 'ec']
df_matched_ic_bc['Data'] = 'ID'
df_ood_ec = df_ood[df_ood['CAM Type'] == 'ec']
df_ood_ec['Data'] = 'OOD'
df_ood_ic_bc = df_ood[df_ood['CAM Type'] != 'ec']
df_ood_ic_bc['Data'] = 'OOD'
df_ec = pd.concat([df_matched_ec, df_ood_ec])
df_ic_bc = pd.concat([df_matched_ic_bc, df_ood_ic_bc])

df_ic = df_ic_bc[df_ic_bc['CAM Type'] == 'ic']
df_bc = df_ic_bc[df_ic_bc['CAM Type'] == 'bc']

cam_type_mapping = {'ic': 'Object', 'bc': 'Edge', 'ec': 'Background'}
view_mapping = {'f': 'Full', 'r': 'Restricted', 'er': 'Extra R.', 'fx': 'Fixed'}
df_ec['CAM Type'] = df_ec['CAM Type'].map(cam_type_mapping)
df_ec['View'] = df_ec['View'].map(view_mapping)
df_ic_bc['CAM Type'] = df_ic_bc['CAM Type'].map(cam_type_mapping)
df_ic_bc['View'] = df_ic_bc['View'].map(view_mapping)
df_ic['CAM Type'] = df_ic['CAM Type'].map(cam_type_mapping)
df_ic['View'] = df_ic['View'].map(view_mapping)
df_bc['CAM Type'] = df_bc['CAM Type'].map(cam_type_mapping)
df_bc['View'] = df_bc['View'].map(view_mapping)

# set categorical order
df_ic['View'] = pd.Categorical(df_ic['View'], categories=order, ordered=True)
df_bc['View'] = pd.Categorical(df_ic['View'], categories=order, ordered=True)
df_ec['View'] = pd.Categorical(df_ic['View'], categories=order, ordered=True)


# --- Compute grouped mean and standard error (SE) for each View × Data ---
# We compute stats separately for Object (ic), Edge (bc), and Background (ec),
# then merge them into a single tidy summary table for convenience.

def _sem(x):
    return x.std(ddof=1) / np.sqrt(x.count())


# Mean/SE for Object CAM (df_ic)
stats_ic = (
    df_ic.groupby(["View", "Data"], as_index=False)["Value"]
    .agg(mean="mean", se=_sem)
    .rename(columns={"mean": "Object_mean", "se": "Object_se"})
)

# Mean/SE for Edge CAM (df_bc)
stats_bc = (
    df_bc.groupby(["View", "Data"], as_index=False)["Value"]
    .agg(mean="mean", se=_sem)
    .rename(columns={"mean": "Edge_mean", "se": "Edge_se"})
)

# Mean/SE for Background CAM (df_ec)
stats_ec = (
    df_ec.groupby(["View", "Data"], as_index=False)["Value"]
    .agg(mean="mean", se=_sem)
    .rename(columns={"mean": "Background_mean", "se": "Background_se"})
)

# Merge into a single table for easy inspection/usage in plots
cam_stats = (
    stats_ic.merge(stats_bc, on=["View", "Data"], how="outer")
    .merge(stats_ec, on=["View", "Data"], how="outer")
    .sort_values(["View", "Data"])  # keep categorical order visually meaningful
)

# Display (or export if needed)
print("\nGrouped mean ± SE by View × Data (Object/Edge/Background):\n")
print(cam_stats)
# If desired, you can also save:
# cam_stats.to_csv("border_cam_group_stats.csv", index=False)

palette = {
    'Object': '#C40F5B',
    'Edge': '#B10026',
    'Background': '#089099',
}


# ---- Object–Background Normalized Contrast (OBNC) ----
# OBNC = (Object - Background) / (Object + Background)
# IMPORTANT: compute OBNC per image BEFORE summary stats, then aggregate.
# We merge the per-image Object (ic) and Background (ec) CAM values on their
# shared identifiers (e.g., image/frame/model/etc., View, and Data), compute
# OBNC per row, and finally take group mean ± SE by View × Data.

# Identify merge keys shared by df_ic and df_ec (excluding the measurement columns)
_shared_cols = [c for c in df_ic.columns if c in df_ec.columns]
_keys = [c for c in _shared_cols if c not in ["Value", "CAM Type"]]

# Prepare frames with explicit column names for clarity
_ic_for_merge = df_ic.rename(columns={"Value": "Object_value"}).drop(columns=["CAM Type"])  # CAM Type is redundant here
_ec_for_merge = df_ec.rename(columns={"Value": "Background_value"}).drop(columns=["CAM Type"])  # CAM Type is redundant here

 # Fast path: column-append since rows are already aligned
if len(_ic_for_merge) != len(_ec_for_merge):
    raise ValueError("Object and Background frames have different leng ths; cannot column-append.")

# Optional lightweight alignment check on keys to avoid silent misalignment
if not _ic_for_merge[_keys].reset_index(drop=True).equals(_ec_for_merge[_keys].reset_index(drop=True)):
    raise ValueError("Object and Background rows are not aligned on keys; cannot column-append.")

ob_merge = _ic_for_merge.copy()
ob_merge["Background_value"] = _ec_for_merge["Background_value"].to_numpy()

# Compute per-image OBNC
with np.errstate(divide='ignore', invalid='ignore'):
    ob_merge["OBNC_value"] = (ob_merge["Object_value"] - ob_merge["Background_value"]) / \
                               (ob_merge["Object_value"] + ob_merge["Background_value"])

# Aggregate OBNC by View × Data to obtain mean and standard error
ratio_df = (
    ob_merge.groupby(["View", "Data"], as_index=False)["OBNC_value"]
    .agg(Value="mean", SE=_sem)
    .sort_values(["View", "Data"],
                 key=lambda s: s.map({v: i for i, v in enumerate(order)}) if s.name == "View" else s)
)

print("\nObject–Background Normalized Contrast (OBNC) by View × Data (per-image aggregation):\n")
print(ratio_df)

# Create figure and subplots (now with 4 panels)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(14.5, 4))

sns.lineplot(
    ax=ax1,
    data=df_ic,
    x="View", y="Value", hue="CAM Type", style='Data', palette=palette,
    markers=True, dashes=True, legend=False, errorbar='se'
)

sns.lineplot(
    ax=ax2,
    data=df_bc,
    x="View", y="Value", hue="CAM Type", style='Data', palette=palette,
    markers=True, dashes=True, legend=False, errorbar='se'
)

sns.lineplot(
    ax=ax3,
    data=df_ec,
    x="View", y="Value", hue="CAM Type", style='Data', palette=palette,
    markers=True, dashes=True, legend=True, errorbar='se'
)

# ---- OBNC panel (right-most) ----
# We'll plot ID as solid and OOD as dashed to match the style of other panels.
# Map categorical x positions for consistent spacing
x_pos = {v: i for i, v in enumerate(order)}
for data_label, style_kwargs in {
    "ID": dict(linestyle="-", marker="o"),
    "OOD": dict(linestyle=(0, (6, 2.25)), marker="X"),
}.items():
    sub = ratio_df[ratio_df["Data"] == data_label]
    xs = [x_pos[v] for v in sub["View"]]
    ys = sub["Value"].to_numpy()
    es = sub["SE"].to_numpy()
    ax4.plot(xs, ys, **style_kwargs, linewidth=1.5, label=data_label, color="#333333")
    ax4.errorbar(xs, ys, yerr=es, fmt="none", capsize=3, linewidth=1.0, color="#333333")

# Ticks and labels use the categorical names
ax4.set_xticks(range(len(order)))
ax4.set_xticklabels(order)
ax4.set_title('Obj.–Back. Ratio')
ax4.set_xlabel('View')
# ax4.set_ylabel('OBNC')
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax4.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax4.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
ax4.set_facecolor('white')
ax4.spines[['top', 'right']].set_visible(False)
for spine in ax4.spines.values():
    spine.set_edgecolor('black')

# Only show data type in legend
handles, labels = ax3.get_legend_handles_labels()
style_labels = df_ec['Data'].unique().astype(str)  # Ensure labels are strings
new_handles_labels = [(h, l) for h, l in zip(handles, labels) if l in style_labels]
new_handles, new_labels = zip(*new_handles_labels)
ax3.legend(new_handles, new_labels, title="Data")

# Rest of the styling remains the same
ax1.grid(False)
ax2.grid(False)
ax3.grid(False)
ax2.set_ylabel('')
ax3.set_ylabel('')
ax2.set_yticks([])

ax1.set_ylim([0.54, 0.75])
ax2.set_ylim([0.54, 0.75])
ax3.set_ylim([0.26, 0.47])
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax3.yaxis.set_major_locator(MaxNLocator(nbins=6))

ax1.set_ylabel('CAM Value')
ax4.set_ylabel('OBR Value')
ax1.set_title('Object')
ax2.set_title('Border')
ax3.set_title('Background')

fig.patch.set_alpha(0)
ax1.set_facecolor('white')
ax2.set_facecolor('white')
ax3.set_facecolor('white')
ax4.set_facecolor('white')

ax1.spines[['top', 'right']].set_visible(False)
ax2.spines[['top', 'right']].set_visible(False)
ax3.spines[['top', 'right']].set_visible(False)
ax4.spines[['top', 'right']].set_visible(False)

# ax1.grid(visible=False, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
# ax2.grid(visible=False, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
# ax3.grid(visible=False, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
ax1.grid(visible=False)
ax2.grid(visible=False)
ax3.grid(visible=False)
ax4.grid(visible=False)

for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(axis="x", which="both", bottom=True, top=False, length=3, width=1, color="black")
    ax.tick_params(axis="y", which="both", left=True, right=False, length=3, width=1, color="black")

for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

plt.savefig('../../plots/border_cam_plot_normal.svg', bbox_inches='tight', format='svg')
