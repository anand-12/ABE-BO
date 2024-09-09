import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats as stats
import textwrap

def process_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        cumulative_regrets = [experiment[3] for experiment in data]  # Index 3 for cumulative regret
        return cumulative_regrets
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

def power_law(x, a, b):
    return a * np.power(x, b)

def fit_sublinearity(x, y):
    popt, _ = curve_fit(power_law, x, y)
    return popt[1]  # Return the exponent

def analyze_regret(regrets):
    sublinearity_orders = []
    for regret in regrets:
        x = np.arange(1, len(regret) + 1)
        sublinearity_order = fit_sublinearity(x, regret)
        sublinearity_orders.append(sublinearity_order)

    mean_sublinearity = np.mean(sublinearity_orders)
    std_error = stats.sem(sublinearity_orders)

    return mean_sublinearity, std_error

def plot_function_scatter(function_name, function_results, x_label_map, method_order):
    plt.figure(figsize=(16, 10))
    means = [function_results[method]['mean'] for method in method_order]
    sems = [function_results[method]['sem'] for method in method_order]

    x_labels = [x_label_map.get(method, method) for method in method_order]
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=15)) for label in x_labels]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    for i, (method, mean, sem) in enumerate(zip(method_order, means, sems)):
        plt.errorbar(i, mean, yerr=sem, fmt='o', capsize=5, capthick=2, color=colors[i], ecolor=colors[i], markersize=10, alpha=0.7, elinewidth=2)

    plt.ylabel("Sublinearity Order", fontsize=18, fontweight='bold')
    plt.title(f"Sublinearity Orders for {function_name}", fontsize=20, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.xticks(range(len(method_order)), wrapped_labels, rotation=0, ha='center', fontsize=12)

    plt.axhline(y=np.mean(means), color='r', linestyle='--', alpha=0.5)
    plt.text(len(method_order)-1, np.mean(means), 'Mean', va='center', ha='left', backgroundcolor='w', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f"sublinearity_scatter_{function_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_relative_heatmap(df_mean, x_label_map, method_order, show_numbers=True):
    df_relative = df_mean.div(df_mean.min(axis=1), axis=0) - 1
    df_relative = df_relative[method_order]  # Reorder columns

    plt.figure(figsize=(20, 16))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 16

    cmap = sns.color_palette("flare", as_cmap=True)
    vmax = np.percentile(df_relative.values, 90)

    if show_numbers:
        sns.heatmap(df_relative, annot=df_mean[method_order].round(3), fmt='.3f', cmap=cmap, 
                    cbar_kws={'label': 'Relative difference from best'},
                    vmin=0, vmax=vmax, annot_kws={"size": 14, "weight": "bold"})
    else:
        sns.heatmap(df_relative, annot=False, cmap=cmap, 
                    cbar_kws={'label': 'Relative difference from best'},
                    vmin=0, vmax=vmax)

    x_labels = [x_label_map.get(col, col) for col in method_order]
    plt.gca().set_xticklabels(x_labels, rotation=45, ha='right', fontsize=16, fontweight='bold')

    plt.yticks(fontsize=16, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('')

    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Relative difference from best', fontsize=16, fontweight='bold')

    plt.tight_layout()
    filename = "relative_sublinearity_heatmap_with_numbers.png" if show_numbers else "relative_sublinearity_heatmap_without_numbers.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def process_files():
    methods = ['base', 'GPHedge_random', 'GPHedge_bandit', 'GPHedge_abe_least_risk']
    method_order = ['base', 'GPHedge_random', 'GPHedge_bandit', 'GPHedge_abe_least_risk']

    x_label_map = {
        'base': 'Standard\nBO',
        'GPHedge_random': 'GPHedge\nRandom',
        'GPHedge_bandit': 'GPHedge\nBandit',
        'GPHedge_abe_least_risk': 'ABE-BO'
    }

    function_results = {}
    all_results = {}

    for file in os.listdir('.'):
        if file.endswith('.npy'):
            function_name = file.split('_')[0]
            method = '_'.join(file.split('_')[1:]).replace('.npy', '')

            if function_name not in function_results:
                function_results[function_name] = {m: {'mean': 0, 'sem': 0} for m in methods}

            if method in methods:
                regrets = process_file(file)
                if regrets:
                    mean_sublinearity, std_error = analyze_regret(regrets)
                    function_results[function_name][method]['mean'] = mean_sublinearity
                    function_results[function_name][method]['sem'] = std_error
                    
                    if function_name not in all_results:
                        all_results[function_name] = {}
                    all_results[function_name][method] = mean_sublinearity

    for function_name, results in function_results.items():
        plot_function_scatter(function_name, results, x_label_map, method_order)

    # Create DataFrame for heatmaps
    df_mean = pd.DataFrame(all_results).T

    # Plot heatmaps
    plot_relative_heatmap(df_mean, x_label_map, method_order, show_numbers=True)
    plot_relative_heatmap(df_mean, x_label_map, method_order, show_numbers=False)

if __name__ == "__main__":
    process_files()
    print("Analysis complete. Scatter plots and heatmaps have been generated.")

# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from scipy.optimize import curve_fit
# import scipy.stats as stats

# def process_file(file_path):
#     try:
#         data = np.load(file_path, allow_pickle=True)
#         cumulative_regrets = [experiment[3] for experiment in data]  # Index 3 for cumulative regret
#         return cumulative_regrets
#     except Exception as e:
#         print(f"Error processing {file_path}: {str(e)}")
#         return []

# def power_law(x, a, b):
#     return a * np.power(x, b)

# def fit_sublinearity(x, y):
#     popt, _ = curve_fit(power_law, x, y)
#     return popt[1]  # Return the exponent

# def analyze_regret(regrets):
#     sublinearity_orders = []
#     for regret in regrets:
#         x = np.arange(1, len(regret) + 1)
#         sublinearity_order = fit_sublinearity(x, regret)
#         sublinearity_orders.append(sublinearity_order)

#     mean_sublinearity = np.mean(sublinearity_orders)
#     std_error = stats.sem(sublinearity_orders)

#     return mean_sublinearity, std_error

# def plot_average_polynomial_order(results):
#     methods = ['base', 'GPHedge_random', 'GPHedge_bandit', 'GPHedge_abe_least_risk']
#     method_labels = ['Standard BO', 'GPHedge Random', 'GPHedge Bandit', 'ABE-BO']
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

#     plt.figure(figsize=(12, 8))

#     x = np.linspace(0, 100, 1000)  # Arbitrary x-axis values

#     for method, label, color in zip(methods, method_labels, colors):
#         orders = [result[method]['mean'] for result in results.values()]
#         mean_order = np.mean(orders)
#         std_error = np.std(orders) / np.sqrt(len(orders))

#         y_mean = x**mean_order
#         y_lower = x**(mean_order - std_error)
#         y_upper = x**(mean_order + std_error)

#         plt.plot(x, y_mean, label=f'{label} (Order: {mean_order:.3f})', color=color)
#         plt.fill_between(x, y_lower, y_upper, alpha=0.2, color=color)

#     plt.xlabel('Iterations', fontsize=14)
#     plt.ylabel('Cumulative Regret', fontsize=14)
#     plt.title('Average Polynomial Order of Cumulative Regret', fontsize=16)
#     plt.legend(fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     # plt.xscale('log')
#     # plt.yscale('log')
    
#     plt.tight_layout()
#     plt.savefig("average_polynomial_order.png", dpi=300, bbox_inches='tight')
#     plt.close()

# def process_files():
#     methods = ['base', 'GPHedge_random', 'GPHedge_bandit', 'GPHedge_abe_least_risk']
#     results = {}

#     for file in os.listdir('.'):
#         if file.endswith('.npy'):
#             function_name = file.split('_')[0]
#             method = '_'.join(file.split('_')[1:]).replace('.npy', '')

#             if method in methods:
#                 if function_name not in results:
#                     results[function_name] = {m: {'mean': 0, 'sem': 0} for m in methods}

#                 regrets = process_file(file)
#                 if regrets:
#                     mean_sublinearity, std_error = analyze_regret(regrets)
#                     results[function_name][method]['mean'] = mean_sublinearity
#                     results[function_name][method]['sem'] = std_error

#     plot_average_polynomial_order(results)

# if __name__ == "__main__":
#     process_files()
#     print("Analysis complete. Average polynomial order plot has been generated.")