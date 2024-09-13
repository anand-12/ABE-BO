import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import OrderedDict

def load_results(file_path):
    return np.load(file_path, allow_pickle=True)

def plot_combined_results(all_file_results, save_path, max_iterations=20, custom_names=None, custom_colors=None):
    fig, ax = plt.subplots(figsize=(15, 10))
    
    
    lines = OrderedDict()
    
    for file_name, results in all_file_results.items():
        all_max_values = np.array([exp[0][:max_iterations] for exp in results])
        median = np.median(all_max_values, axis=0)
        lower = np.percentile(all_max_values, 25, axis=0)
        upper = np.percentile(all_max_values, 75, axis=0)
        
        label = custom_names.get(file_name, file_name) if custom_names else file_name
        color = custom_colors.get(label, 'blue')  
        
        line, = ax.plot(range(max_iterations), median, label=label, linewidth=3, color=color)
        ax.fill_between(range(max_iterations), lower, upper, alpha=0.3, color=color)
        
        
        lines[label] = line

    
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    
    
    
    
    ax.set_xlim(0, max_iterations - 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    
    ordered_lines = [lines[name] for name in custom_names.values()]
    ax.legend(ordered_lines, custom_names.values(), loc='lower right', fontsize=35, bbox_to_anchor=(0.99, 0.01), ncol=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'all_files_comparison_percentile_ranges_{max_iterations}_iterations.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    results_dir = "."
    save_dir = "./Visualizations_CartPole"
    os.makedirs(save_dir, exist_ok=True)
    
    custom_names = OrderedDict([
        ("CartPole_GPHedge_base.npy", "Standard BO"),
        ("CartPole_GPHedge_random.npy", "Random Portfolio"),
        ("CartPole_GPHedge_bandit.npy", "GP-Hedge"),
        ("CartPole_GPHedge_abe_least_risk.npy", "ABE-BO")
    ])
    
    
    custom_colors = {
        "Standard BO": "#00cfff", 
        "GP-Hedge": "#428050",    
        "Random Portfolio": "#fff158",   
        "ABE-BO": "#ff0057"       
    }
    
    all_file_results = {}
    for file_name in os.listdir(results_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(results_dir, file_name)
            results = load_results(file_path)
            all_file_results[file_name] = results
    
    plot_combined_results(all_file_results, save_dir, max_iterations=50, custom_names=custom_names, custom_colors=custom_colors)
    print(f"Combined visualization for all files saved in {save_dir}")
    print("All visualizations completed.")