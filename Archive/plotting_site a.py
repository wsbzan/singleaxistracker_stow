import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


CSV_CANDIDATES = [
    'site_stow_conditions.csv',
    'Final Version/Output/site_a_stow_conditions.csv',
    'tracker_angles_with_stow_conditions.csv',
]


def find_existing_csv(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f'Could not find any expected CSV file. Tried: {candidates}'
    )


def find_column(df, candidates):
    normalized = {c.strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def add_trigger_background(ax, data, time_col, trigger_col, trigger_colors):
    if data.empty:
        return

    current_trigger = data[trigger_col].iloc[0]
    start_idx = 0

    for i in range(1, len(data)):
        if data[trigger_col].iloc[i] != current_trigger:
            ax.axvspan(
                data[time_col].iloc[start_idx],
                data[time_col].iloc[i],
                alpha=0.25,
                color=trigger_colors[current_trigger],
            )
            current_trigger = data[trigger_col].iloc[i]
            start_idx = i

    ax.axvspan(
        data[time_col].iloc[start_idx],
        data[time_col].iloc[-1],
        alpha=0.25,
        color=trigger_colors[current_trigger],
    )


csv_path = find_existing_csv(CSV_CANDIDATES)
df = pd.read_csv(csv_path)

timestamp_col = find_column(df, ['timestamp_local', 'Timestamp (UTC)', 'timestamp', 'td'])
trigger_col = find_column(df, ['trigger'])
stow_col = find_column(df, ['stow_angle', 'stow angle'])
pos1_col = find_column(df, ['pos 1', 'pos_1', 'pos1'])

if timestamp_col is None:
    raise KeyError('Could not find a timestamp column.')
if trigger_col is None:
    raise KeyError('Could not find trigger column.')
if stow_col is None:
    raise KeyError('Could not find stow angle column.')
if pos1_col is None:
    raise KeyError("Could not find 'pos 1' column in CSV.")

df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)
df[trigger_col] = df[trigger_col].ffill().fillna('Unknown')

# Requested date ranges (inclusive by date)
date_ranges = [
    ('2022-01-16', '2022-01-17'),
    ('2023-12-31', '2024-01-01'),
]

filtered_sets = []
for start_date, end_date in date_ranges:
    start_ts = pd.Timestamp(start_date)
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    mask = (df[timestamp_col] >= start_ts) & (df[timestamp_col] < end_exclusive)
    filtered_sets.append(df.loc[mask].copy())

all_triggers = pd.concat(filtered_sets)[trigger_col].dropna().unique() if any(len(s) for s in filtered_sets) else ['Unknown']
colors = plt.cm.Set3(np.linspace(0, 1, len(all_triggers)))
trigger_colors = {trigger: colors[i] for i, trigger in enumerate(all_triggers)}

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

for ax, (start_date, end_date), subset in zip(axes, date_ranges, filtered_sets):
    if subset.empty:
        ax.set_title(f'{start_date} to {end_date}\n(no data)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Angle (degrees)')
        ax.grid(alpha=0.3)
        continue

    add_trigger_background(ax, subset, timestamp_col, trigger_col, trigger_colors)

    ax.plot(
        subset[timestamp_col],
        subset[stow_col],
        label='Stow Angle',
        linewidth=2,
        color='black',
    )
    ax.plot(
        subset[timestamp_col],
        subset[pos1_col],
        label='Pos 1',
        linewidth=2,
        color='blue',
    )

    ax.set_title(f'{start_date} to {end_date}')
    ax.set_xlabel('Time')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

axes[0].set_ylabel('Angle (degrees)')

trigger_legend_patches = [
    Patch(facecolor=trigger_colors[t], alpha=0.5, label=t) for t in all_triggers
]
fig.legend(
    handles=trigger_legend_patches,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.06),
    ncol=min(len(trigger_legend_patches), 5),
    title='Triggers',
)
fig.legend(
    handles=[
        Line2D([0], [0], color='black', linewidth=2, label='Stow Angle'),
        Line2D([0], [0], color='blue', linewidth=2, label='Pos 1'),
    ],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.0),
    ncol=2,
)

fig.suptitle('Stow Angle vs Pos 1 with Trigger Background', y=1.12)
fig.tight_layout()
plt.savefig('stow_angle_vs_pos1_split_ranges.png', dpi=150, bbox_inches='tight')