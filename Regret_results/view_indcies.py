import numpy as np
import matplotlib.pyplot as plt
import os


color_palette = ['

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
    'GPHedge_random': 'Random\nPortfolio',
    'base': 'Standard\nBO'
}

def load_data(function_name):
    data = {}
    
    for method_name in legend_order:
        file_path = f"{function_name}_{method_name}.npy"
        if os.path.exists(file_path):
            method_data = np.load(file_path, allow_pickle=True)
            af_selections = method_data[0][-2]  
            data[method_name] = af_selections
    
    return data

def plot_acquisition_function_heatmap(data, af_options, custom_labels, function_name):
    fig, ax = plt.subplots(figsize=(18, 10))  
    y_positions = []
    y_ticks = []
    y_pos = 0
    for method in legend_order:
        if method in data:
            selections = data[method]
            heatmap_data = np.array([af_options.index(s) if s in af_options else -1 for s in selections])
            scatter = ax.scatter(range(len(selections)), [y_pos] * len(selections),
                                c=heatmap_data, cmap=plt.cm.get_cmap('viridis', len(af_options)),
                                marker='s', s=600, vmin=-0.5, vmax=len(af_options)-0.5)  
            ax.text(-10, y_pos, custom_labels.get(method, method), ha='right', va='center', fontsize=30)
            y_positions.append(y_pos)
            y_ticks.append(y_pos)
            y_pos -= 2  

    ax.set_yticks([])
    ax.set_xticks(range(0, len(selections), 10))
    ax.set_xticklabels(range(0, len(selections), 10), fontsize=24)
    
    ax.set_xlim(-12, len(selections) + 2)
    ax.set_ylim(y_pos + 1, 1)  

    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.7) 
    cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.03])  
    cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks(range(len(af_options)))
    cbar.set_ticklabels(af_options)
    cbar.ax.tick_params(labelsize=20)
    

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.95)  

    
    filename = f"{function_name}_acquisition_functions_heatmap.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved as {filename}")

function_name = "Hartmann"  
data = load_data(function_name)
af_options = ['LogEI', 'LogPI', 'UCB_0.1', 'UCB_0.3', 'UCB_0.7', 'UCB_0.9']

plot_acquisition_function_heatmap(data, af_options, method_name_mapping, function_name)