import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

df = pd.read_csv('april 2 and 3 example.csv')
df['trigger'] = df['trigger'].ffill()  # Fill NaN values forward

# Convert timestamp to datetime for proper indexing
df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])

# Create a mapping of unique triggers to colors
unique_triggers = df['trigger'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(unique_triggers)))
trigger_colors = {trigger: colors[i] for i, trigger in enumerate(unique_triggers)}

plt.figure(figsize=(12, 6))

# Add background colored regions for each zone/trigger
current_trigger = df['trigger'].iloc[0]
start_idx = 0

for i in range(1, len(df)):
    if df['trigger'].iloc[i] != current_trigger:
        # Plot a vertical span for the current zone
        plt.axvspan(df['timestamp_local'].iloc[start_idx], 
                   df['timestamp_local'].iloc[i], 
                   alpha=0.2, 
                   color=trigger_colors[current_trigger])
        current_trigger = df['trigger'].iloc[i]
        start_idx = i

# Don't forget the last zone
plt.axvspan(df['timestamp_local'].iloc[start_idx], 
           df['timestamp_local'].iloc[-1], 
           alpha=0.2, 
           color=trigger_colors[current_trigger])

# Plot main data on top
# plt.plot(df['timestamp_local'], df['stow_angle'], label='Stow Angle', linewidth=2)
plt.plot(df['timestamp_local'], df['tracker_theta'], label='Tracker Theta', linewidth=2)
plt.plot(df['timestamp_local'], df['Wind Dir (Deg)'], label='Wind Dir (Deg)', linewidth=2, linestyle='--')

# Create legend for zones
legend_patches = [Patch(facecolor=trigger_colors[trigger], alpha=0.2, label=trigger) 
                  for trigger in unique_triggers]
plt.legend(handles=legend_patches, loc='upper left', title='Zones')

# Create legend for main data on the right
plt.legend(['Stow Angle', 'Tracker Theta', 'Wind Dir (Deg)'], loc='center right')

plt.xlabel('Timestamp (Local)')
plt.ylabel('Angle (degrees)')
plt.title('Tracker Angles Over Time with Zone Background')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('stow_angle_over_time.png', dpi=150)