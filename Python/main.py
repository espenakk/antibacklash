import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the data
file_path = 'Python/data/megalog2.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Prepare results file
log_name = os.path.splitext(os.path.basename(file_path))[0]
output_filename = f"Python/data/results_from_{log_name}.txt"

# Convert timestamp to datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Get unique TestIndex values
test_indices = df['TestIndex'].unique()

# Store performance scores, validity status and results for logging
performance_scores = {}
test_validity = {}
all_test_results = []

WITHIN_PERCENTAGE = 10

ANTIBACKLASH_MODE_MAP = {
    0: "Adaptive torque",
    1: "Constant torque",
    2: "Speed ref offset",
    3: "Slave drooping",
    4: "Position offset",
    5: "Speed ref delay",
    6: "Speed ref delay offset",
    7: "Simple torque",
    8: "Position speed offset"
}

def generate_plot(df_test, test_index, backlash_results, performance_score=None):
    # Create a new figure for each test index with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 22), 
                                           gridspec_kw={'height_ratios': [3, 3, 3, 1.5]})
    
    title = f'Antibacklash Performance Analysis - Test: {round(test_index)}'
    if performance_score is not None:
        title += f' (Score: {performance_score:.2f}%)'
    fig.suptitle(title, fontsize=16)

    # --- Speed Plot ---
    ax1.plot(df_test['Time (s)'], df_test['SpeedRef'], label='SpeedRef')
    ax1.plot(df_test['Time (s)'], df_test['ENC1Speed'], label='ENC1Speed')
    ax1.plot(df_test['Time (s)'], df_test['FC1Speed'], label='FC1Speed')
    ax1.plot(df_test['Time (s)'], df_test['FC2Speed'], label='FC2Speed')
    ax1.set_ylabel('Speed')
    ax1.grid(True)
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

    # Add backlash lines to plots
    for res in backlash_results:
        ax1.axvline(x=res['start'], color='g', linestyle='--')
        ax1.axvline(x=res['end'], color='r', linestyle='--')

    # Update legend for speed plot
    handles, labels = ax1.get_legend_handles_labels()
    if backlash_results:
        if 'Backlash Start' not in labels:
            handles.append(plt.Line2D([0], [0], color='g', linestyle='--', label='Backlash Start'))
        if 'Backlash End' not in labels:
            handles.append(plt.Line2D([0], [0], color='r', linestyle='--', label='Backlash End'))
    by_label = dict(zip([h.get_label() for h in handles], handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left')

    # --- Add vertical line for AntiBacklashEnabled switch ---
    switch_point = df_test[df_test['AntiBacklashEnabled'].diff() == 1]
    
    if not switch_point.empty:
        switch_time = switch_point['Time (s)'].iloc[0]
        
        # Add a vertical line to each plot
        ax1.axvline(x=switch_time, color='m', linestyle='-', linewidth=1, label='AntiBacklash ON')
        ax2.axvline(x=switch_time, color='m', linestyle='-', linewidth=1)
        ax3.axvline(x=switch_time, color='m', linestyle='-', linewidth=1)

        # Update legend for the speed plot to include the new line
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper left')

    # --- Parameters Table ---
    ax4.axis('off') # Hide the axes for the table subplot
    param_names = [
        'Offset', 'BaseTorque', 'GainTorque',
        'LoadTorque', 'MaxTorque', 'SlaveDroop', 'MasterDroop',
        'SlaveDelay', 'DegreeOffset', 'DegreeGain', 'AntiBacklashMode'
    ]
    # Get parameters for the specific test_index, assuming they are constant for the test
    param_values = [df_test[param].iloc[0] for param in param_names]

    table_data = []
    for name, val in zip(param_names, param_values):
        if name == 'AntiBacklashMode':
            table_data.append([name, ANTIBACKLASH_MODE_MAP.get(val, f"Unknown ({val})")])
        else:
            table_data.append([name, val])
            
    table = ax4.table(cellText=table_data, colLabels=['Parameter', 'Value'],
                      loc='center', cellLoc='center', colWidths=[0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

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
    
    # get results
    df_test['SpeedError'] = df_test['ENC1Speed'] - df_test['SpeedRef']
    mean_abs_error = df_test['SpeedError'].abs().mean()
    within_pcs = (df_test['SpeedError'].abs() <= df_test['SpeedRef'].abs() * WITHIN_PERCENTAGE/100).mean() * 100


    # --- Test Validity Check ---
    # A test is invalid if ENC1Speed never reaches 90% of SpeedRef (positive and negative) while AntiBacklash is enabled.
    is_valid = True
    # Check the part of the test where AntiBacklash is enabled
    if df_test['AntiBacklashEnabled'].any():
        df_ab_enabled = df_test[df_test['AntiBacklashEnabled'] == 1]
        if not df_ab_enabled.empty:
            # Check for positive speed reference
            max_pos_speed_ref = df_ab_enabled[df_ab_enabled['SpeedRef'] > 0]['SpeedRef'].max()
            if pd.notna(max_pos_speed_ref) and max_pos_speed_ref > 0:
                max_pos_enc1_speed = df_ab_enabled['ENC1Speed'].max()
                if max_pos_enc1_speed < max_pos_speed_ref * 0.90:
                    is_valid = False
                    print(f"Info: Test {round(test_index)} is invalid because ENC1Speed did not reach positive SpeedRef while AntiBacklash was enabled.")

            # Check for negative speed reference, only if still valid
            if is_valid:
                min_neg_speed_ref = df_ab_enabled[df_ab_enabled['SpeedRef'] < 0]['SpeedRef'].min()
                if pd.notna(min_neg_speed_ref) and min_neg_speed_ref < 0:
                    min_neg_enc1_speed = df_ab_enabled['ENC1Speed'].min()
                    if min_neg_enc1_speed > min_neg_speed_ref * 0.90: # min_neg_speed_ref is negative, so > means smaller in magnitude
                        is_valid = False
                        print(f"Info: Test {round(test_index)} is invalid because ENC1Speed did not reach negative SpeedRef while AntiBacklash was enabled.")
    test_validity[test_index] = is_valid

    # --- Backlash Analysis ---
    # Define intervals starting each time SpeedRef changes polarity (sign flips),
    # ignoring transitions that only go to 0.
    # Each interval has fixed length of 1.5 seconds.
    backlash_results = []
    last_nonzero_sign = None
    last_nonzero_index = None  # keep for potential future use
    sign_series = np.sign(df_test['SpeedRef'].to_numpy())
    times = df_test['Time (s)'].to_numpy()

    def add_interval(start_sample_index: int):
        t_start_local = times[start_sample_index]
        t_end_target_local = t_start_local + 1.5
        # Fast search for end index (first time > target) then step one back
        end_idx_local = np.searchsorted(times, t_end_target_local, side='right') - 1
        if end_idx_local < start_sample_index:
            return
        t_end_local = times[end_idx_local]
        duration_local = t_end_local - t_start_local
        if duration_local <= 0:
            return
        pos_start_local = df_test['ENC1Position'].iloc[start_sample_index]
        pos_end_local = df_test['ENC1Position'].iloc[end_idx_local]
        movement_local = abs(pos_end_local - pos_start_local)
        interval_df_local = df_test.iloc[start_sample_index:end_idx_local+1].copy()
        dt_local = interval_df_local['Time (s)'].diff().fillna(0)
        iae_local = (interval_df_local['SpeedError'].abs() * dt_local).sum()
        ise_local = ((interval_df_local['SpeedError']**2) * dt_local).sum()
        speed_ref_start = df_test['SpeedRef'].iloc[start_sample_index]
        driving_motor_pos_col = 'FC1Position' if speed_ref_start >= 0 else 'FC2Position'
        motor_pos_start = df_test[driving_motor_pos_col].iloc[start_sample_index]
        motor_pos_end = df_test[driving_motor_pos_col].iloc[end_idx_local]
        motor_delta = abs(motor_pos_end - motor_pos_start)
        encoder_delta = movement_local  # already abs ENC1 position change
        backlash_deg = max(0.0, motor_delta - encoder_delta)
        backlash_results.append({
            'start': t_start_local,
            'end': t_end_local,
            'duration': duration_local,
            'movement': movement_local,
            'IAE': iae_local,
            'ISE': ise_local,
            'backlash_deg': backlash_deg,
            'motor_delta': motor_delta,
            'encoder_delta': encoder_delta,
            'driving_motor': driving_motor_pos_col
        })

    for idx, s in enumerate(sign_series):
        if s == 0:
            continue  # ignore zeros for polarity change detection
        if last_nonzero_sign is None:
            add_interval(idx)  # first movement interval
            last_nonzero_sign = s
            last_nonzero_index = idx
            continue
        if s != last_nonzero_sign:
            add_interval(idx)
            last_nonzero_sign = s
            last_nonzero_index = idx
        else:
            last_nonzero_sign = s
            last_nonzero_index = idx

    # --- Log Results & Performance Score ---
    if backlash_results:
        # Warn if the number of events is not 16
        if len(backlash_results) != 16:
            print(f"Warning: Test {round(test_index)} has {len(backlash_results)} backlash events, expected 16.")

        # Get parameters for logging
        param_names_for_log = [
            'AntiBacklashMode', 'Offset', 'BaseTorque', 'GainTorque',
            'LoadTorque', 'MaxTorque', 'SlaveDroop', 'MasterDroop',
            'SlaveDelay', 'DegreeOffset', 'DegreeGain'
        ]
        param_values_for_log = {name: df_test[name].iloc[0] for name in param_names_for_log}

        current_score = None
        # Calculate performance score if we have 16 events and the test is valid
        if len(backlash_results) >= 16 and is_valid:
            event_diff_sums = []
            for ev in backlash_results[:16]:  # only first 16 matter for score
                start_t = ev['start']
                end_t = ev['end']
                ev_df = df_test[(df_test['Time (s)'] >= start_t) & (df_test['Time (s)'] <= end_t)]
                if ev_df.empty:
                    event_diff_sums.append(0.0)
                    continue
                pos_mask = ev_df['SpeedRef'] > 0
                neg_mask = ev_df['SpeedRef'] < 0
                pos_sum = (ev_df.loc[pos_mask, 'ENC1Speed'] - ev_df.loc[pos_mask, 'FC1Speed']).abs().sum()
                neg_sum = (ev_df.loc[neg_mask, 'ENC1Speed'] - ev_df.loc[neg_mask, 'FC2Speed']).abs().sum()
                diff_sum = pos_sum + neg_sum
                ev['diff_sum'] = diff_sum  # store for potential logging
                event_diff_sums.append(diff_sum)

            sum1_8 = sum(event_diff_sums[0:8])
            sum9_16 = sum(event_diff_sums[8:16])
            if sum1_8 > 0:
                performance_score = (sum9_16 / sum1_8) * 100.0
                performance_scores[test_index] = performance_score
                current_score = performance_score

        iae_list = [e['IAE'] for e in backlash_results]
        ise_list = [e['ISE'] for e in backlash_results]
        backlash_deg_list = [e.get('backlash_deg', 0.0) for e in backlash_results]
        mean_backlash_first8 = np.mean(backlash_deg_list[0:8]) if len(backlash_deg_list) >= 8 else np.nan
        mean_backlash_last8 = np.mean(backlash_deg_list[8:16]) if len(backlash_deg_list) >= 16 else np.nan
        backlash_reduction_pct = None
        if not np.isnan(mean_backlash_first8) and mean_backlash_first8 > 0 and not np.isnan(mean_backlash_last8):
            backlash_reduction_pct = (mean_backlash_last8 / mean_backlash_first8) * 100.0

        mean_iae = np.mean(iae_list)
        max_iae = np.max(iae_list)
        mean_ise = np.mean(ise_list)
        max_ise = np.max(ise_list)
        
        all_test_results.append({
            'test_index': test_index,
            'is_valid': is_valid,
            'params': param_values_for_log,
            'backlash_results': backlash_results,
            'performance_score': current_score,
            'df_test': df_test,
            'mean_abs_error': mean_abs_error,
            'within_pcs': within_pcs,
            "mean_iae": mean_iae,
            'max_iae': max_iae,
            'mean_ise': mean_ise,
            "max_ise": max_ise,
            'mean_backlash_first8': mean_backlash_first8,
            'mean_backlash_last8': mean_backlash_last8,
            'backlash_reduction_pct': backlash_reduction_pct
        })


# --- Generate Final Summary ---
summary_string = ""
best_test_index = None
if performance_scores:
    # Find the test with the best (lowest) performance score
    best_test_index = min(performance_scores, key=performance_scores.get)
    best_score = performance_scores[best_test_index]

    acceptable_tests = [test for test, score in performance_scores.items() if score < 100]
    unacceptable_tests = [test for test, score in performance_scores.items() if score >= 100]
    
    num_scored_tests = len(performance_scores)
    num_acceptable = len(acceptable_tests)
    
    percentage_acceptable = (num_acceptable / num_scored_tests) * 100 if num_scored_tests > 0 else 0

    summary_string += "--- Performance Summary ---\n"
    summary_string += f"Best performance was in Test {round(best_test_index)} with a score of {best_score:.2f}%.\n"
    summary_string += "A score under 100% is acceptable. Lower is better.\n"
    summary_string += f"\n{num_acceptable} out of {num_scored_tests} scored tests were acceptable ({percentage_acceptable:.1f}%).\n"

    if unacceptable_tests:
        summary_string += "\nThe following tests had unacceptable scores (>= 100%):\n"
        # Sort unacceptable tests by their score
        unacceptable_tests_sorted = sorted(unacceptable_tests, key=lambda x: performance_scores[x])
        for test in unacceptable_tests_sorted:
            summary_string += f"  - Test {round(test)}: {performance_scores[test]:.2f}%\n"
else:
    summary_string += "\nNo valid tests were found to calculate performance scores.\n"

invalid_test_indices = [idx for idx, valid in test_validity.items() if not valid and idx != 0]
if invalid_test_indices:
    summary_string += "\nThe following tests were disqualified for not meeting speed requirements:\n"
    for test_index in sorted(invalid_test_indices):
        summary_string += f"  - Test {round(test_index)}\n"

# Print summary to console
print("\n" + summary_string)

# --- Write Sorted Log File ---
# Sort results: valid tests by score, then invalid tests at the bottom
# all_test_results.sort(key=lambda x: (not x['is_valid'], x['performance_score'] if x['performance_score'] is not None else float('inf')))
all_test_results.sort(key=lambda x: (x['performance_score'] if x['performance_score'] is not None else float('inf')))

# changed to encoding=utf-8
with open(output_filename, 'w', encoding='utf-8') as f:
    f.write(f"Backlash Analysis Results for {file_path}\n\n")
    f.write(summary_string + "\n\n")

    for result in all_test_results:
        result_string = f"--- Test: {round(result['test_index'])} ---\n"
        if not result['is_valid']:
            result_string += "Status: INVALID (ENC1Speed did not reach SpeedRef)\n"

        if len(result['backlash_results']) < 16:
            result_string += "Status: INVALID (Did not find 16 backlash events)\n"
        
        result_string += "Parameters:\n"
        for name, val in result['params'].items():
            if name == 'AntiBacklashMode':
                result_string += f"  {name}: {ANTIBACKLASH_MODE_MAP.get(val, f'Unknown ({val})')}\n"
            else:
                result_string += f"  {name}: {val}\n"
        result_string += "\n"

        result_string += f"Detected Backlash Events: {len(result['backlash_results'])}\n"
        
        if result['performance_score'] is not None:
            result_string += f"Performance Score: {result['performance_score']:.2f}%\n"
            if result['performance_score'] < 100:
                result_string += "Result: Acceptable\n"
            else:
                result_string += "Result: Not Acceptable\n"
        elif len(result['backlash_results']) == 31:
             result_string += "Performance Score: N/A (sum of first 8 events is zero)\n"

        result_string += f"\nMean abs speed error: {result['mean_abs_error']:.2f}\n"
        result_string += f"% time speed within \u00B1{WITHIN_PERCENTAGE}%: {result['within_pcs']:.1f}\n"
        result_string += f"Mean IAE: {result['mean_iae']:.3f}\n"
        result_string += f" Max IAE: {result['max_iae']:.3f}\n"
        result_string += f"Mean ISE: {result['mean_ise']:.3f}\n"
        result_string += f" Max ISE: {result['max_ise']:.3f}\n"
        if 'mean_backlash_first8' in result and not np.isnan(result['mean_backlash_first8']):
            result_string += f"Mean Backlash First 8: {result['mean_backlash_first8']:.3f}°\n"
        if 'mean_backlash_last8' in result and not np.isnan(result['mean_backlash_last8']):
            result_string += f"Mean Backlash Last 8: {result['mean_backlash_last8']:.3f}°\n"
        if result.get('backlash_reduction_pct') is not None:
            result_string += f"Backlash Reduction % (last/first*100): {result['backlash_reduction_pct']:.2f}%\n"
     
        result_string += "\nBacklash Events:\n"
        for i, res in enumerate(result['backlash_results']):
            extra = ""
            if 'diff_sum' in res:
                extra = f", diff_sum = {res['diff_sum']:.3f}"
            backlash_info = ''
            if 'backlash_deg' in res:
                backlash_info = f", backlash = {res['backlash_deg']:.3f}°"
            result_string += (
                f"  Event {i+1:2d}: Duration = {res['duration']:.4f}s, Δposition = {res['movement']:.3f}°"
                f" (from {res['start']:6.2f}s to {res['end']:6.2f}s){backlash_info}{extra}\n"
            )
        
        f.write(result_string + "\n")

# --- Interactive Plotting ---
while True:
    print("\n--- Plotting Options ---")
    print("1. Show plot for the BEST test")
    print("2. Show plots for all ACCEPTABLE tests")
    print("3. Show plots for all INVALID tests")
    print("4. Show plots for all UNACCEPTABLE tests")
    print("5. Show plots for ALL tests")
    print("6. Show plots for TOP 20 tests")
    print("0. Exit")
    
    choice = input("Enter your choice: ")

    if choice == '1':
        if best_test_index is not None:
            test_to_plot = next((r for r in all_test_results if r['test_index'] == best_test_index), None)
            if test_to_plot:
                generate_plot(test_to_plot['df_test'], test_to_plot['test_index'], test_to_plot['backlash_results'], test_to_plot['performance_score'])
        else:
            print("No best test found to plot.")
    elif choice == '2':
        tests_to_plot = [r for r in all_test_results if r['is_valid'] and r['performance_score'] is not None and r['performance_score'] < 100]
        for test in tests_to_plot:
            generate_plot(test['df_test'], test['test_index'], test['backlash_results'], test['performance_score'])
    elif choice == '3':
        tests_to_plot = [r for r in all_test_results if not r['is_valid']]
        for test in tests_to_plot:
            generate_plot(test['df_test'], test['test_index'], test['backlash_results'], test['performance_score'])
    elif choice == '4':
        tests_to_plot = [r for r in all_test_results if r['is_valid'] and r['performance_score'] is not None and r['performance_score'] >= 100]
        for test in tests_to_plot:
            generate_plot(test['df_test'], test['test_index'], test['backlash_results'], test['performance_score'])
    elif choice == '5':
        for test in all_test_results:
            generate_plot(test['df_test'], test['test_index'], test['backlash_results'], test['performance_score'])
    elif choice == '6':
        top_tests = sorted(
            [r for r in all_test_results if r['performance_score'] is not None],
            key=lambda x: x['performance_score']
        )[:20]
        if not top_tests:
            print("No scored tests available.")
        else:
            print(f"Plotting {len(top_tests)} best overall tests.")
            for test in top_tests:
                generate_plot(test['df_test'], test['test_index'], test['backlash_results'], test['performance_score'])
    elif choice == '0':
        break
    else:
        print("Invalid choice. Please try again.")
        continue

    # Show plots if any were generated
    if plt.get_fignums():
        plt.show()