import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

df = pd.read_csv('april 2 and 3 example.csv')
df['trigger'] = df['trigger'].ffill()  # Fill NaN values forward

# Convert timestamp to datetime for proper indexing
df['timestamp_local'] = pd.to_datetime(df['timestamp_local'], format='%-m/%-d/%y %-H:%M')

# Create a mapping of unique triggers to colors
unique_triggers = df['trigger'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(unique_triggers)))
colors = ['gray', 'blue', 'red', 'green', 'cyan', 'magenta'][:len(unique_triggers)]  # Use a predefined color list
trigger_colors = {trigger: colors[i] for i, trigger in enumerate(unique_triggers)}

fig, ax = plt.subplots(figsize=(12, 6))
# plt.figure(figsize=(12, 6))

# Add background colored regions for each zone/trigger
current_trigger = df['trigger'].iloc[0]
start_idx = 0

for i in range(1, len(df)):
    if df['trigger'].iloc[i] != current_trigger:
        # Plot a vertical span for the current zone
        plt.axvspan(df['timestamp_local'].iloc[start_idx], 
                   df['timestamp_local'].iloc[i], 
                   alpha=0.3, 
                   color=trigger_colors[current_trigger])
        current_trigger = df['trigger'].iloc[i]
        start_idx = i

# Don't forget the last zone
plt.axvspan(df['timestamp_local'].iloc[start_idx], 
           df['timestamp_local'].iloc[-1], 
           alpha=0.3, 
           color=trigger_colors[current_trigger])

# Plot main data on top
line1, = ax.plot(df['timestamp_local'], df['stow_angle'], label='Stow Angle', linewidth=2, color='black')
line2, = ax.plot(df['timestamp_local'], df['tracker_theta'], label='Tracker Theta', linewidth=2, color='blue')
ax2 = ax.twinx()  # Create a secondary y-axis for the additional variables
line3, = ax2.plot(df['timestamp_local'], df['Wind Gust Speed (m/s)'], label='Wind Gust Speed (m/s)', linewidth=2, linestyle='--', color = 'orange', alpha =0.5)
ax3 = ax.twinx()  # Create another secondary y-axis for the additional variables
line4, = ax3.plot(df['timestamp_local'], df['Snowfall Rate (mm/hr)'], label='Snowfall Rate (mm/hr)', linewidth=2, linestyle=':', color = 'purple', alpha =0.5)
ax4 = ax.twinx()  # Create another secondary y-axis for the additional variables
line5, = ax4.plot(df['timestamp_local'], df['Wind Dir (Deg)'], label='Wind Direction (Deg)', linewidth=2, linestyle='-.', color = 'green', alpha =0.5)
# Create legend for zones
legend_patches = [Patch(facecolor=trigger_colors[trigger], alpha=0.5, label=trigger) 
                  for trigger in unique_triggers]
first_legend = ax.legend(handles=legend_patches, loc='upper left', title='Zones')

plt.legend([line1, line2, line3, line4, line5],
            ['Stow Angle', 'Ideal Angle', 'Wind Gust Speed', 'Snowfall Rate', 'Wind Direction'],
            loc='lower right')

ax.add_artist(first_legend)  # Add the first legend back to the plot

ax3.spines['right'].set_position(('axes', 1.05))  # Move the third y-axis further to the right
ax4.spines['right'].set_position(('axes', 1.1))  #

plt.xlabel('Local Time')
plt.ylabel('Angle (degrees)')
# plt.title('Tracker Angles Over Time with Zone Background')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()


ax2.set_ylabel('Wind Gust Speed (m/s)', color='orange')
ax3.set_ylabel('Snowfall Rate (mm/hr)', color='purple')
ax4.set_ylabel('Wind Direction (Deg)', color='green')
plt.savefig('stow_angle_over_time.png', dpi=150)