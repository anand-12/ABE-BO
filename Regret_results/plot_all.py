import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
from matplotlib.lines import Line2D
import argparse
from fnmatch import fnmatch


color_palette = ['#e41a1c', '#ff0057', '#428050', '#fff158', '#00cfff', '#ffff33', '#a65628', '#f781bf', '#41f221', '#00CED1']

method_styles = {
    'GPHedge_abe_least_risk': {'color': color_palette[1]},
    'GPHedge_bandit': {'color': color_palette[2]},
    'GPHedge_random': {'color': color_palette[3]},
    'base': {'color': color_palette[4]}
}
legend_order = ['base', 'GPHedge_random', 'GPHedge_bandit', 'GPHedge_abe_least_risk']


method_name_mapping = {
    'GPHedge_abe_least_risk': 'ABE-BO',
    'GPHedge_bandit': 'GP-Hedge',
    'GPHedge_random': 'Random Portfolio',
    'base': 'Standard BO'
}


def process_file(file_path):
    data = np.load(file_path, allow_pickle=True)
    metrics = []
    for experiment in data:
        experiment_metrics = [experiment[i] for i in range(4)]  
        metrics.append(experiment_metrics)
    return np.array(metrics)


def create_adaptive_legend(ax, lines, labels):
    legend_elements = []
    
    
    line_dict = {label: line for line, label in zip(lines, labels)}
    
    
    for method in legend_order:
        if method in line_dict:
            line = line_dict[method]
            mapped_label = method_name_mapping.get(method, method)
            legend_elements.append(Line2D([0], [0], color=line.get_color(), lw=4, label=mapped_label))
    
    
    ax.legend(handles=legend_elements, loc='upper right',
              ncol=1, borderaxespad=1, fontsize='xx-large')

def plot_metrics(file_paths, output_dir, function_name):
    all_data = [process_file(file_path) for file_path in file_paths]
    file_names = [os.path.basename(file_path).replace('.npy', '').split('_', 1)[1] for file_path in file_paths]
    metric_names = ["Maximum Value", "Gap Metric", "Simple Regret", "Cumulative Regret"]
    iterations = range(1, len(all_data[0][0][0]) + 1)

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('default')
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    for i, metric in enumerate(metric_names):
        fig, ax = plt.subplots(figsize=(10, 7))
        lines = []
        for file_data, file_name in zip(all_data, file_names):
            style = method_styles.get(file_name, {'color': 'gray'})
            
            
            median = np.median(file_data, axis=0)[i]
            
            
            if function_name.lower().startswith('hartmann'):
                lower = np.percentile(file_data, 40, axis=0)[i]
                upper = np.percentile(file_data, 60, axis=0)[i]
            else:
                lower = np.percentile(file_data, 25, axis=0)[i]
                upper = np.percentile(file_data, 75, axis=0)[i]
            
            
            line, = ax.plot(iterations, median, color=style['color'], linewidth=2.5, label=file_name)
            lines.append(line)
            
            
            error_x = iterations[::10]
            ax.errorbar(error_x, median[::10], 
                        yerr=[median[::10] - lower[::10], upper[::10] - median[::10]],
                        fmt='none', ecolor=style['color'], capsize=5, alpha=0.5)

        if metric in ["Simple Regret"]:
            ax.set_yscale('log')
        
        
        ax.tick_params(axis='both', which='major', labelsize=25, width=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        create_adaptive_legend(ax, lines, file_names)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric.replace(" ", "_")}_{function_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Plots for {function_name} have been saved in the '{output_dir}' directory.")
def process_datasets(base_path, ignore_patterns):
    function_files = {}
    for file in glob.glob(os.path.join(base_path, '*.npy')):
        full_file_name = os.path.basename(file)
        function_name = full_file_name.split('_')[0]
        file_name = full_file_name.replace('.npy', '').split('_', 1)[1]
        
        
        if any(fnmatch(full_file_name, pattern) for pattern in ignore_patterns):
            print(f"Ignoring file: {full_file_name}")
            continue
        
        if function_name not in function_files:
            function_files[function_name] = []
        function_files[function_name].append(file)

    for function_name, file_paths in function_files.items():
        output_dir = os.path.join(base_path, 'metric_plots_abe', function_name)
        plot_metrics(file_paths, output_dir, function_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and plot metrics data.")
    parser.add_argument("--base_path", default=".", help="Base path for data files")
    parser.add_argument("--ignore", nargs="*", default=[], help="List of file patterns to ignore (e.g., '*_GPHedge_random.npy')")
    
    args = parser.parse_args()
    
    print(f"Ignoring files matching the following patterns: {args.ignore}")
    process_datasets(args.base_path, args.ignore)