import csv
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

RISING_THRESHOLD = 0.005
FILENAME = 'log 3.csv'
WINDOW = 7  # odd numbers: 5,7,11 -> larger=more smoothing

showAllPlotsAndPrints = True

def to_float(s):
    s = s.strip()
    return float(s) if s else math.nan

def first_rise_index(values, thr, id, id_list):
    for i in range(id, len(values)):
        if math.isnan(values[i]) or math.isnan(values[i-1]):
            continue
        if values[i] - values[i-1] > thr:
            id_list.append(i)
            return i, True
    return None, False

def first_crossing_index(ref, act, start, id_list):
    for i in range(start, len(ref)):
        if math.isnan(ref[i]) or math.isnan(act[i]):
            continue
        if act[i-1] < ref[i-1] and act[i] >= ref[i] and ref[i] == ref[i+1]:
            id_list.append(i)
            return i, False
    return None, True

def moving_average(x, window_size=5):
    w = np.ones(int(window_size), dtype=float) / window_size
    return np.convolve(x, w, mode='same')

def fill_nans(x):
    isn = np.isnan(x)
    if isn.any():
        good = ~isn
        x[isn] = np.interp(np.flatnonzero(isn), np.flatnonzero(good), x[good])
    return x

def check_interval(speedRef_data, encoder_data, antiBacklashEnabled, interval_idx):
    interval_start = interval_idx[0]
    interval_stop = interval_idx[-1]
    
    t_interval = t_s[interval_start: interval_stop]
    speedRef_data = speedRef_data[interval_start: interval_stop]
    encoder_data = encoder_data[interval_start: interval_stop]
    antiBacklashEnabled = antiBacklashEnabled[interval_start: interval_stop]

    start_id_list, match_id_list = [], []
    foundFirstIndex = False
    last_match_id = 0
    for _ in t_dt:
        if foundFirstIndex:
            last_match_id, foundFirstIndex = first_crossing_index(speedRef_data, encoder_data, start_idx, match_id_list)
        else:
            start_idx, foundFirstIndex = first_rise_index(speedRef_data, RISING_THRESHOLD, last_match_id+1, start_id_list)

    # remove false matches, might need adjustment if test results get a lot better
    filtered_starts, filtered_matches = [], []
    for s, m in zip(start_id_list, match_id_list):
        if abs(s-m) > 5 and (t_dt[m] - t_dt[s]).total_seconds() > 0.2:
            filtered_starts.append(s)
            filtered_matches.append(m)
    start_id_list = filtered_starts
    match_id_list = filtered_matches

    ti_slice = [testIndex[i] for i in interval_idx][0]
    delta_s_comparison = []
    for s, m in zip(start_id_list, match_id_list):
        delta_s = (t_dt[m] - t_dt[s]).total_seconds()
        delta_s_comparison.append(delta_s)

    return start_id_list, match_id_list, t_interval, speedRef_data, encoder_data, delta_s_comparison, ti_slice

def plotAndPrintAll(start_id_list, match_id_list, t_interval, speedRef_data, encoder_data, interval_idx):
    delta_s_comparison = []
    count = 0

    for s, m in zip(start_id_list, match_id_list):
        delta_s = (t_dt[m] - t_dt[s]).total_seconds()
        delta_s_comparison.append(delta_s)
        count+=1

        if count == (len(start_id_list)/2 + 1):
            print("-----------------------------------------\n")

        if antiBacklashEnabled[s] == '1':
            print("Antibacklash enabled")
        else:
            print("Antibacklash not enabled")
        
        print(f"Rise starts at   {(t_dt[s] - t0).total_seconds()} (idx {s})")
        print(f"Equal at top at  {(t_dt[m] - t0).total_seconds()} (idx {m})")
        print(f"\u0394t = {delta_s:.3f} s\n")

    print("-----------------------------------------\n")
    print(f"Test 1: {delta_s_comparison[0]-delta_s_comparison[4]:.3f} s difference")
    print(f"Test 2: {delta_s_comparison[1]-delta_s_comparison[5]:.3f} s difference")
    print(f"Test 3: {delta_s_comparison[2]-delta_s_comparison[6]:.3f} s difference")
    print(f"Test 4: {delta_s_comparison[3]-delta_s_comparison[7]:.3f} s difference\n")

    _, ax = plt.subplots(figsize=(16, 8))
    ax.plot(t_interval, speedRef_data, label="Reference speed")
    ax.plot(t_interval, encoder_data, label="Scaled encoder speed")
    for s, m in zip(start_id_list, match_id_list):
        ax.axvline(t_interval[s], color='green', linestyle='--', linewidth=1, label='Start')
        ax.axvline(t_interval[m], color='red', linestyle='--', linewidth=1, label='Match')
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper left')
    ax.set_ylabel("Speed (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.grid(which='major', linestyle='-', linewidth=0.8, color='grey')
    ax.grid(which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    plt.title(f"Speed ref vs encoder speed (test: {testIndex[interval_idx[0]]})")
    plt.tight_layout()
    plt.show()
    plt.close()

def plotAndPrintBest():
    pass
    #ikke ferdig, ønsker funksjon som plotter bare beste dataene

(
    t_str,
    t_dt,
    speedRef,
    enc,
    FC1Speed,
    FC2Speed,
    FC1Torque,
    FC2Torque,
    FC3Torque,
    encPosition,
    antiBacklashEnabled,
    offset,
    baseTorque,
    gainTorque,
    loadTorque,
    maxTorque,
    slaveDroop,
    masterDroop,
    running,
    testIndex
) = [ [] for _ in range(20) ]

with open("data/"+FILENAME, newline='') as file:
    reader = csv.reader(file)
    header = next(reader)

    for row in reader:
        if not row or not row[0].strip():
            continue

        dt = datetime.fromisoformat(row[0].strip())
        t_dt.append(dt)
        t_str.append(dt.strftime('%H:%M:%S.%f')[:-3])

        speedRef.append(to_float(row[1]))
        enc.append(to_float(row[2]))
        FC1Speed.append(to_float(row[3]))
        FC2Speed.append(to_float(row[4]))
        FC1Torque.append(row[5])
        FC2Torque.append(row[6])
        FC3Torque.append(row[7])
        encPosition.append(row[8])
        antiBacklashEnabled.append(row[9])
        offset.append(row[10])
        baseTorque.append(row[11])
        gainTorque.append(row[12])
        loadTorque.append(row[13])
        maxTorque.append(row[14])
        slaveDroop.append(row[15])
        masterDroop.append(row[16])
        running.append(row[17])
        testIndex.append(row[18])

t0 = t_dt[0]
t_s = np.array([(dt - t0).total_seconds() for dt in t_dt])

speedRef = fill_nans(np.array(speedRef))
speedRef = moving_average(speedRef, WINDOW)

# find the interval we want to look at
# run the test, find the delta times
# save best test times and averages for each interval, compare at the end
# print useful data to terminal

intervals = []
in_test = False
for i in range(1, len(running)):
    if running[i] == '1' and not in_test:
        in_test = True
        start = i
    elif running[i] != '1' and in_test:
        end = i
        intervals.append((start, end))
        in_test = False

if in_test:
    intervals.append((start, len(running)))

best_delta_times = []
interval_averages = []

for start, end in intervals:
    idx_list = list(range(start, end))
    start_id_list, match_id_list, t_interval, speedRef_data, encoder_data, delta_s_comparison, ti_slice = check_interval(speedRef, enc, antiBacklashEnabled, idx_list)
    best_d = min(delta_s_comparison)
    avg_d = np.mean(delta_s_comparison)
    best_delta_times.append((best_d, ti_slice))
    interval_averages.append((avg_d, ti_slice))
    if showAllPlotsAndPrints:
        plotAndPrintAll(start_id_list, match_id_list, t_interval, speedRef_data, encoder_data, idx_list)

best_d, best_ti = min(best_delta_times, key=lambda x: x[0])   
best_avg, best_avg_ti = min(interval_averages, key=lambda x: x[0])

# enda ikke implementert
if showAllPlotsAndPrints == False:
    plotAndPrintBest()

print(f"Best \u0394t is {best_d:.3f} s, happening in test number: {best_ti}")
print(f"Best average \u0394t is {best_avg:.3f} s, happening in test number: {best_avg_ti}")