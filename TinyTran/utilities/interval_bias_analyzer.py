import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import numpy as np
from scipy.stats import skew
from composer_melody_trainer import build_dataset, INTERVAL_VOCAB

def analyze_interval_bias(meta_csv, midi_base, composer_filter):
    # Override global variables directly if required by build_dataset
    from composer_melody_trainer import META_CSV, MIDI_BASE, COMPOSER
    META_CSV = meta_csv
    MIDI_BASE = midi_base
    COMPOSER = composer_filter

    sequences = build_dataset()
    interval_sums = []

    for seed, target_intervals, *_ in sequences:
        total_sum = sum(target_intervals)
        interval_sums.append(total_sum)

    if not interval_sums:
        print(f"No sequences found for composer '{composer_filter}' in dataset.")
        return

    # Calculate Statistics
    interval_sums_np = np.array(interval_sums)
    mean_sum = np.mean(interval_sums_np)
    median_sum = np.median(interval_sums_np)
    p25 = np.percentile(interval_sums_np, 25)
    p75 = np.percentile(interval_sums_np, 75)
    skewness_val = skew(interval_sums_np)

    # Plot Histogram
    plt.hist(interval_sums_np, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of Interval Sums - Composer: {composer_filter}")
    plt.xlabel("Total Interval Sum (Net Pitch Change Over Sequence)")
    plt.ylabel("Number of Sequences")
    plt.grid(True)
    plt.show()

    # Print Statistics
    print(f"\n🎼 Composer: {composer_filter}")
    print(f"📈 Average Total Interval Sum: {mean_sum:.2f}")
    print(f"📏 Median Total Interval Sum: {median_sum:.2f}")
    print(f"📊 25th Percentile: {p25:.2f}")
    print(f"📊 75th Percentile: {p75:.2f}")
    print(f"🎯 Skewness: {skewness_val:.2f}")
    print(f"📚 Total Sequences Analyzed: {len(interval_sums)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Interval Bias in Training Data")
    parser.add_argument("--composer", type=str, required=True, help="Composer name to filter by (e.g., Mozart, Chopin)")
    parser.add_argument("--meta_csv", type=str, required=True, help="Path to annotated metadata CSV")
    parser.add_argument("--midi_base", type=str, required=True, help="Base path to MIDI files")
    args = parser.parse_args()

    analyze_interval_bias(args.meta_csv, args.midi_base, args.composer)