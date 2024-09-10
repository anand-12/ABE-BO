import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
npy_files_path = '/Users/anand/Desktop/ABE-BO/Results_abe/NPY_files/'
print("NPY files directory exists:", os.path.exists(npy_files_path))
print("Contents of NPY files directory:", os.listdir(npy_files_path))
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
    npy_files_path = '/Users/anand/Desktop/ABE-BO/Results_abe/NPY_files/'
    print(f"Searching for .npy files in: {npy_files_path}")
    
    for file in os.listdir(npy_files_path):
        if file.endswith('.npy'):
            file_path = os.path.join(npy_files_path, file)
            print(f"Processing file: {file}")
            
            if 'base' in file.lower():
                data_dict['base'].extend(process_file(file_path))
            elif 'gphedge_random' in file.lower():
                data_dict['GPHedge_random'].extend(process_file(file_path))
            elif 'gphedge_bandit' in file.lower():
                data_dict['GPHedge_bandit'].extend(process_file(file_path))
            elif 'gphedge_abe_least_risk' in file.lower():
                data_dict['ABE'].extend(process_file(file_path))
            else:
                print(f"File {file} doesn't match any category")
    
    print("Data collected:")
    for key, value in data_dict.items():
        print(f"{key}: {len(value)} data points")
    
    if all(len(v) == 0 for v in data_dict.values()):
        print("No data was collected. Check file naming or processing.")
        return
    
    df = pd.DataFrame({k: pd.Series(v) for k, v in data_dict.items()})
    print("DataFrame created with shape:", df.shape)
    
    x_label_map = { 
        'base': 'Standard\nBO',
        'GPHedge_random': 'MA\nRandom',
        'GPHedge_bandit': 'MA\nBandit',
        'ABE': 'ABE-BO'
    }
    
    plt.figure(figsize=(10, 8))  # Adjusted figure size
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    n_colors = len(data_dict)
    palette = sns.color_palette("rocket", n_colors)
    sns.set_palette(palette)
    
    try:
        ax = sns.boxplot(data=df, orient='v', showfliers=False, width=0.6, order=data_dict.keys())
        print("Boxplot created successfully")
    except Exception as e:
        print(f"Error creating boxplot: {str(e)}")
        return
    
    # plt.ylabel("Final Gap Metric", fontsize=22, fontweight='bold')
    # plt.xlabel("")
    
    # Use x_label_map for x-axis labels
    ax.set_xticklabels([x_label_map[label.get_text()] for label in ax.get_xticklabels()],
                       rotation=0, ha='center', fontsize=18, fontweight='bold')
    
    ax.tick_params(axis='y', labelsize=18, labelcolor='#555555')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    
    try:
        plt.savefig('ensemble_gap_metric_boxplot.png', dpi=300, bbox_inches='tight')
        print("Plot saved successfully as 'ensemble_gap_metric_boxplot.png'")
    except Exception as e:
        print(f"Error saving plot: {str(e)}")
    
    plt.close()

if __name__ == "__main__":
    plot_ensemble_gap_metric()