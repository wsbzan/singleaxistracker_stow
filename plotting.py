import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

x = ['Final Version/Output/site_example_no_zero_no_preemptive_stow_stow_conditions.csv',
     'Final Version/Output/site_example_no_zero_preemptive_stow_conditions.csv',
     'Final Version/Output/site_example_zero_no_preemptive_stow_conditions.csv',
     'Final Version/Output/site_example_zero_preemptive_stow_conditions.csv']

# Fixed per-trigger colors so the same trigger always renders the same color across all charts.
TRIGGER_COLORS = {
    'Ideal Tracking': 'gray',
    'Relaxing': 'purple',
    'Storm': 'green',
    'Wind': 'red',
    'Snow': 'blue',
    'Preemptive Storm': 'cyan',
    'Preemptive Snow': 'yellow',
}
FALLBACK_COLOR = 'black'

for xx in x:
    df = pd.read_csv(xx)
    df['trigger'] = df['trigger'].ffill()  # Fill NaN values forward

    # Convert timestamp to datetime for proper indexing (CSV uses ISO-style "YYYY-MM-DD HH:MM:SS")
    df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])

    # Filter to April 2 and 3 2024
    df = df[(df['timestamp_local'] >= '2024-04-02') & (df['timestamp_local'] < '2024-04-04')]

    # Map triggers present in this file to their fixed colors (unseen triggers fall back).
    unique_triggers = df['trigger'].unique()
    trigger_colors = {trigger: TRIGGER_COLORS.get(trigger, FALLBACK_COLOR) for trigger in unique_triggers}

    fig, ax = plt.subplots(figsize=(10, 5))
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
    first_legend = ax.legend(handles=legend_patches, loc='upper left', title='Zones',
                             fontsize=9)

    plt.legend([line1, line2, line3, line4, line5],
                ['Stow Angle', 'Ideal Angle', 'Wind Gust Speed', 'Snowfall Rate', 'Wind Direction'],
                loc='lower right',
                fontsize=9)

    ax.add_artist(first_legend)  # Add the first legend back to the plot

    ax3.spines['right'].set_position(('axes', 1.07))  # Move the third y-axis further to the right
    ax4.spines['right'].set_position(('axes', 1.132))  #

    ax.set_xlabel('Local Time')
    ax.set_ylabel('Angle (degrees)')
    # plt.title('Tracker Angles Over Time with Zone Background')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ax2.set_ylabel('Wind Gust Speed (m/s)', color='orange')
    ax3.set_ylabel('Snowfall Rate (mm/hr)', color='purple')
    ax4.set_ylabel('Wind Direction (Deg)', color='green')

    plt.savefig(f'stow_angle_over_time_{xx.split("/")[-1].replace(".csv", "")}.png', dpi=145, bbox_inches='tight')
    plt.close()  # Close the current figure to avoid overlap with the next iteration