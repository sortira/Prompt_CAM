import argparse
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np


def calculate_mean_auc(csv_path, plot_output=None):
    """Calculate mean AUC from insertion scores CSV."""
    auc_scores = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    auc = float(row['auc'])
                    auc_scores.append(auc)
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping invalid row: {row}")
                    continue
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    if not auc_scores:
        print("Error: No valid AUC scores found in CSV")
        sys.exit(1)
    
    mean_auc = sum(auc_scores) / len(auc_scores)
    
    print(f"Total images: {len(auc_scores)}")
    print(f"Mean Insertion AUC: {mean_auc:.6f}")
    print(f"Min AUC: {min(auc_scores):.6f}")
    print(f"Max AUC: {max(auc_scores):.6f}")
    
    # Create histogram plot
    if plot_output:
        plt.figure(figsize=(10, 6))
        plt.hist(auc_scores, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(mean_auc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_auc:.4f}')
        plt.xlabel('Insertion AUC Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Insertion AUC Scores (n={len(auc_scores)})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_output, bbox_inches='tight', dpi=150)
        print(f"Plot saved to: {plot_output}")
        plt.close()
    
    return mean_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mean AUC from insertion scores CSV")
    parser.add_argument("csv_path", type=str, help="Path to the insertion_scores.csv file")
    parser.add_argument("--plot", type=str, default="insertion_auc_distribution.png", help="Output path for histogram plot")
    args = parser.parse_args()
    
    calculate_mean_auc(args.csv_path, args.plot)
