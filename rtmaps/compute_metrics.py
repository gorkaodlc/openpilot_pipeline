import json
import numpy as np
import pandas as pd
import argparse

def process_json(file_path, output_path):
    # Load the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)
    
    # Organize results by range
    ranges = ["0-10m", "10-20m", "20-30m", "30-40m", "+40m"]
    stats_by_range = {r: {"mean_error": [], "max_error": [], "std_error": [], "AP@0.5": [], "AP@1": [], "AP@2": []} for r in ranges}

    # Populate stats_by_range dictionary
    for entry in data["errors_by_range"]:
        for r in ranges:
            stats_by_range[r]["mean_error"].append(entry[r]["mean_error"])
            stats_by_range[r]["max_error"].append(entry[r]["max_error"])
            stats_by_range[r]["std_error"].append(entry[r]["std_error"])
            stats_by_range[r]["AP@0.5"].append(entry[r]["hit_rates"]["AP@0.5"])
            stats_by_range[r]["AP@1"].append(entry[r]["hit_rates"]["AP@1"])
            stats_by_range[r]["AP@2"].append(entry[r]["hit_rates"]["AP@2"])

    # Compute required statistics for each range
    results = {}
    for r in ranges:
        results[r] = {
            "mean_mean_error": np.mean(stats_by_range[r]["mean_error"]),
            "max_max_error": np.max(stats_by_range[r]["max_error"]),
            "mean_std_error": np.mean(stats_by_range[r]["std_error"]),
            "mean_AP@0.5": np.mean(stats_by_range[r]["AP@0.5"]),
            "mean_AP@1": np.mean(stats_by_range[r]["AP@1"]),
            "mean_AP@2": np.mean(stats_by_range[r]["AP@2"])
        }
    
    df_results = pd.DataFrame.from_dict(results, orient="index")
    print(df_results)
    
    # Save results to a CSV file
    df_results.to_csv(output_path, index=True)
    print(f"Results saved to {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file and compute statistics.")
    parser.add_argument("file_path", type=str, help="Path to the JSON file")
    parser.add_argument("output_path", type=str, help="Path to save the results CSV file")
    args = parser.parse_args()
    
    process_json(args.file_path, args.output_path)
