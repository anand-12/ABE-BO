import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def process_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        final_gap_metrics = [experiment[1][-1] for experiment in data]
        return np.array(final_gap_metrics)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return np.array([])

def plot_ensemble_gap_metric():
    data_dict = {
        'base': [],
        'GPHedge_random': [],
        'GPHedge_bandit': [],
        'ABE': []  # This will contain the least risk ABE variant
    }

    for file in os.listdir('.'):
        if file.endswith('.npy'):
            file_path = os.path.join('.', file)
            if 'base' in file.lower():
                data_dict['base'].extend(process_file(file_path))
            elif 'gphedge_random' in file.lower():
                data_dict['GPHedge_random'].extend(process_file(file_path))
            elif 'gphedge_bandit' in file.lower():
                data_dict['GPHedge_bandit'].extend(process_file(file_path))
            elif 'gphedge_abe_least_risk' in file.lower():
                data_dict['ABE'].extend(process_file(file_path))

    df = pd.DataFrame({k: pd.Series(v) for k, v in data_dict.items()})

    y_label_map = {
        'base': 'Standard\nBO',
        'GPHedge_random': 'GPHedge\nRandom',
        'GPHedge_bandit': 'GPHedge\nBandit',
        'ABE': 'ABE-BO'
    }

    plt.figure(figsize=(8, 10))  # Adjusted figure size
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)

    n_colors = len(data_dict)
    palette = sns.color_palette("rocket", n_colors)
    sns.set_palette(palette)

    ax = sns.boxplot(data=df, orient='h', showfliers=False, width=0.6, order=data_dict.keys())
    plt.xlabel("Final Gap Metric", fontsize=22, fontweight='bold')
    plt.ylabel("")

    ax.set_yticklabels([y_label_map.get(label.get_text(), label.get_text()) for label in ax.get_yticklabels()],
                       rotation=0, va='center', fontsize=18, fontweight='bold')

    for tick in ax.get_yticklabels():
        tick.set_x(-0.01)

    # plt.axhline(y=0.5, color='#777777', linestyle='--', alpha=0.7, linewidth=2)
    # plt.axhline(y=1.5, color='#777777', linestyle='--', alpha=0.7, linewidth=2)

    # plt.text(plt.xlim()[1] * 1.02, 0, 'Baseline', va='center', ha='left', fontsize=20, fontweight='bold', color='#555555', rotation=-90)
    # plt.text(plt.xlim()[1] * 1.02, 2, 'GPHedge Variants', va='center', ha='left', fontsize=20, fontweight='bold', color='#555555', rotation=-90)

    ax.tick_params(axis='x', labelsize=18, labelcolor='#555555')
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig('ensemble_gap_metric_boxplot_rotated.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Rotated box plot has been saved as 'ensemble_gap_metric_boxplot_rotated.png'.")

if __name__ == "__main__":
    plot_ensemble_gap_metric()