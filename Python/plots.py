import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'data/log.csv' 
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Convert timestamp to datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Get unique TestIndex values
test_indices = df['TestIndex'].unique()

# --- Plotting for each TestIndex ---
for test_index in test_indices:
    # Skip TestIndex 0
    if test_index == 0:
        continue

    # Filter dataframe for the current TestIndex and when the test is running
    df_test = df[(df['TestIndex'] == test_index) & (df['Running'] == 1)].copy()
    
    if df_test.empty:
        continue
    
    # Calculate elapsed time in seconds for each test run
    df_test['Time (s)'] = (df_test['Timestamp'] - df_test['Timestamp'].iloc[0]).dt.total_seconds()

    # Create a new figure for each test index with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 22), 
                                           gridspec_kw={'height_ratios': [3, 3, 3, 1.5]})
    fig.suptitle(f'Antibacklash Performance Analysis - Test: {round(test_index)}', fontsize=16)

    # --- Speed Plot ---
    ax1.plot(df_test['Time (s)'], df_test['SpeedRef'], label='SpeedRef')
    ax1.plot(df_test['Time (s)'], df_test['ENC1Speed'], label='ENC1Speed')
    ax1.plot(df_test['Time (s)'], df_test['FC1Speed'], label='FC1Speed')
    ax1.plot(df_test['Time (s)'], df_test['FC2Speed'], label='FC2Speed')
    ax1.set_ylabel('Speed')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    ax1.set_title('Speed vs. Time')

    # --- Position Plot ---
    ax2.plot(df_test['Time (s)'], df_test['ENC1Position'], label='ENC1Position')
    ax2.plot(df_test['Time (s)'], df_test['FC1Position'], label='FC1Position')
    ax2.plot(df_test['Time (s)'], df_test['FC2Position'], label='FC2Position')
    ax2.plot(df_test['Time (s)'], df_test['FC3Position'], label='FC3Position')
    ax2.set_ylabel('Position')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    ax2.set_title('Position vs. Time')

    # --- Torque Plot ---
    ax3.plot(df_test['Time (s)'], abs(df_test['FC1Torque']), label='FC1Torque')
    ax3.plot(df_test['Time (s)'], abs(df_test['FC2Torque']), label='FC2Torque')
    ax3.plot(df_test['Time (s)'], df_test['FC3Torque'], label='FC3Torque')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Torque')
    ax3.grid(True)
    ax3.legend(loc='upper left')
    ax3.set_title('Torque vs. Time')

    # --- Add vertical line for AntiBacklashEnabled switch ---
    switch_point = df_test[df_test['AntiBacklashEnabled'].diff() == 1]
    
    if not switch_point.empty:
        switch_time = switch_point['Time (s)'].iloc[0]
        
        # Add a vertical line to each plot
        ax1.axvline(x=switch_time, color='r', linestyle='--', label='AntiBacklash On')
        ax2.axvline(x=switch_time, color='r', linestyle='--')
        ax3.axvline(x=switch_time, color='r', linestyle='--')
        
        # Update legend for the speed plot to include the new line
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper left')

    # --- Parameters Table ---
    ax4.axis('off') # Hide the axes for the table subplot
    param_names = [
        'Offset', 'BaseTorque', 'GainTorque',
        'LoadTorque', 'MaxTorque', 'SlaveDroop', 'MasterDroop',
        'AntiBacklashMode'
    ]
    # Get parameters for the specific test_index
    param_values = [df_test[param].iloc[0] for param in param_names]

    table_data = [[name, val] for name, val in zip(param_names, param_values)]
    table = ax4.table(cellText=table_data, colLabels=['Parameter', 'Value'],
                      loc='center', cellLoc='center', colWidths=[0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.show()