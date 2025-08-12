import os
import sys
import csv

OLD_COLUMNS = [
    "Timestamp","SpeedRef","ENC1Speed","FC1Speed","FC2Speed",
    "ENC1Position","FC1Position","FC2Position","FC3Position",
    "FC1Torque","FC2Torque","FC3Torque",
    "AntiBacklashEnabled","Offset","BaseTorque","GainTorque",
    "LoadTorque","MaxTorque","SlaveDroop","MasterDroop",
    "Running","TestIndex","AntiBacklashMode"
]

NEW_COLUMNS = [
    "Timestamp","SpeedRef","ENC1Speed","FC1Speed","FC2Speed",
    "ENC1Position","FC1Position","FC2Position","FC3Position",
    "FC1Torque","FC2Torque","FC3Torque",
    "AntiBacklashEnabled","Offset","BaseTorque","GainTorque",
    "LoadTorque","MaxTorque","SlaveDroop","MasterDroop",
    "SlaveDelay","DegreeOffset","DegreeGain",
    "Running","TestIndex","AntiBacklashMode"
]

def detect_format(header_cols):
    if header_cols == NEW_COLUMNS:
        return "new"
    if header_cols == OLD_COLUMNS:
        return "old"
    return "unknown"

def normalize_row_to_new(old_row):
    # Fill missing empty fields with 0
    old_row = ["0" if (c is None or c == '') else c for c in old_row]
    # Insert three zeros before last three (Running, TestIndex, AntiBacklashMode)
    # OLD layout indices: ... MasterDroop (index 19), Running (20), TestIndex (21), AntiBacklashMode (22)
    # We want: ... MasterDroop, SlaveDelay, DegreeOffset, DegreeGain, Running, TestIndex, AntiBacklashMode
    extended = old_row[:20] + ["0","0","0"] + old_row[20:]
    # Normalize -0
    extended = ["0" if v in ("-0","+0") else v for v in extended]
    return extended

def convert_file(in_path, out_path, overwrite=False):
    if (not overwrite) and os.path.exists(out_path):
        print(f"Skip (exists): {out_path}")
        return

    with open(in_path, newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"Empty file: {in_path}")
            return

        fmt = detect_format(header)
        if fmt == "new":
            # Copy through, also fix empty cells to 0 if desired
            rows = []
            for row in reader:
                row = ["0" if c == '' else c for c in row]
                rows.append(row)
            with open(out_path, 'w', newline='') as w:
                writer = csv.writer(w)
                writer.writerow(NEW_COLUMNS)
                writer.writerows(rows)
            print(f"Copied (already new): {in_path} -> {out_path}")
            return
        if fmt != "old":
            print(f"ERROR: Unexpected columns in {in_path}")
            return

        # Process old -> new
        rows = []
        for row in reader:
            if not row or all(c.strip()=='' for c in row):
                continue
            # Pad/truncate to 23 just in case
            if len(row) < len(OLD_COLUMNS):
                row += [''] * (len(OLD_COLUMNS)-len(row))
            row = row[:len(OLD_COLUMNS)]
            rows.append(normalize_row_to_new(row))

    with open(out_path, 'w', newline='') as w:
        writer = csv.writer(w)
        writer.writerow(NEW_COLUMNS)
        writer.writerows(rows)
    print(f"Converted: {in_path} -> {out_path} ({len(rows)} data rows)")

def main():
    import argparse
    p = argparse.ArgumentParser(description="Convert old antibacklash CSV logs to new format.")
    p.add_argument("input", help="Input file or directory")
    p.add_argument("output", help="Output file or directory")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = p.parse_args()

    in_path = args.input
    out_path = args.output

    if os.path.isdir(in_path):
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
        for root, _, files in os.walk(in_path):
            rel_root = os.path.relpath(root, in_path)
            target_root = out_path if rel_root == "." else os.path.join(out_path, rel_root)
            os.makedirs(target_root, exist_ok=True)
            for f in files:
                if f.lower().endswith(".csv"):
                    src = os.path.join(root, f)
                    dst = os.path.join(target_root, f)
                    convert_file(src, dst, overwrite=args.overwrite)
    else:
        if os.path.isdir(out_path):
            dst = os.path.join(out_path, os.path.basename(in_path))
        else:
            dst = out_path
        convert_file(in_path, dst, overwrite=args.overwrite)

if __name__ == "__main__":
    main()