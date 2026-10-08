"""Assignment 03: summarize a telemetry ward's systolic readings.

Run from the assignment directory with the project environment active:

    python3 analysis.py
"""

import numpy as np


def load_readings(filename):
    """Return (patient_ids, monitors, hour_columns, readings) from the supplied CSV.

    readings is a 2D array of integers: one row per patient, one column per
    monitored hour, in the order the header lists them.
    """
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    header = lines[0].strip().split(",")
    rows = [line.strip().split(",") for line in lines[1:] if line.strip()]

    patient_ids = np.array([row[0] for row in rows])
    monitors = np.array([row[1] for row in rows])
    hour_columns = np.array(header[2:])
    readings = np.array([row[2:] for row in rows]).astype(int)
    return patient_ids, monitors, hour_columns, readings


def main():
    patient_ids, monitors, hour_columns, readings = load_readings("data/bp_readings.csv")
    print(f"Loaded {readings.shape[0]} patients x {readings.shape[1]} hours")

    patient_means = readings.mean(axis=1)
    hour_means = readings.mean(axis=0)
    highest = patient_means.argmax()
    peak = hour_means.argmax()
    stage_2 = patient_means >= 140

    names = sorted(set(monitors))
    averages = []
    for name in names:
        averages.append(patient_means[monitors == name].mean())
    high = np.array(averages).argmax()

    in_highest = monitors == names[high]
    others = ~in_highest
    other_mean = patient_means[others].mean()

    print(f"Highest monitor: {names[high]} ({averages[high]:.1f} mmHg)")
    print(f"Other monitors: {others.sum()} patients, mean {other_mean:.1f} mmHg")
    print(f"Difference: {averages[high] - other_mean:.1f} mmHg")

    with open("output/vitals_summary.txt", "w") as file:
        file.write(f"patients: {readings.shape[0]}\n")
        file.write(f"readings: {readings.size}\n")
        file.write(f"mean_sbp: {readings.mean():.2f}\n")
        file.write(f"sd_sbp: {readings.std():.2f}\n")
        file.write(f"min_sbp: {readings.min()}\n")
        file.write(f"max_sbp: {readings.max()}\n")
        file.write(f"stage2_patients: {stage_2.sum()}\n")
        file.write(f"highest_patient: {patient_ids[highest]}\n")
        file.write(f"highest_patient_mean: {patient_means[highest]:.2f}\n")
        file.write(f"peak_hour_column: {hour_columns[peak]}\n")
        file.write(f"peak_hour_mean: {hour_means[peak]:.2f}\n")
        file.write(f"high_monitor: {names[high]}\n")
        file.write(f"monitor_offset: {averages[high] - other_mean:.2f}\n")
        file.write(f"stage2_other_monitors: {(stage_2 & others).sum()}\n")
if __name__ == "__main__":
    main()