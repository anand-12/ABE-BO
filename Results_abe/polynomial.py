import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import scipy.stats as stats

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

def plot_average_polynomial_order(results, custom_names=None):
    methods = ['base', 'GPHedge_random', 'GPHedge_bandit', 'GPHedge_abe_least_risk']
    default_labels = ['Standard BO', 'GPHedge Random', 'GPHedge Bandit', 'ABE-BO']
    
    # Use custom names if provided, otherwise use default labels
    method_labels = custom_names if custom_names else default_labels
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    plt.figure(figsize=(12, 8))

    x = np.linspace(0, 100, 1000)  # Arbitrary x-axis values

    for method, label, color in zip(methods, method_labels, colors):
        orders = [result[method]['mean'] for result in results.values()]
        mean_order = np.mean(orders)
        std_error = np.std(orders) / np.sqrt(len(orders))

        y_mean = x**mean_order
        y_lower = x**(mean_order - std_error)
        y_upper = x**(mean_order + std_error)

        plt.plot(x, y_mean, label=f'{label} (Order: {mean_order:.3f})', color=color)
        plt.fill_between(x, y_lower, y_upper, alpha=0.2, color=color)

    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("average_polynomial_order.png", dpi=300, bbox_inches='tight')
    plt.close()
def process_files():
    methods = ['base', 'GPHedge_random', 'GPHedge_bandit', 'GPHedge_abe_least_risk']
    results = {}
    
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Use absolute path
    directory = '/Users/anand/Desktop/SBU/ICASSP2025/Results_abe/NPY_files'
    print(f"Looking for files in: {directory}")
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return
    
    # List all files in the directory
    all_files = os.listdir(directory)
    print(f"All files in directory: {all_files}")
    
    # Filter for .npy files
    npy_files = [f for f in all_files if f.endswith('.npy')]
    print(f"NPY files found: {npy_files}")
    
    for file in npy_files:
        file_path = os.path.join(directory, file)
        print(f"Processing file: {file_path}")
        
        function_name = file.split('_')[0]
        method = '_'.join(file.split('_')[1:]).replace('.npy', '')
        
        if method in methods:
            if function_name not in results:
                results[function_name] = {m: {'mean': 0, 'sem': 0} for m in methods}
            
            regrets = process_file(file_path)
            if regrets:
                mean_sublinearity, std_error = analyze_regret(regrets)
                results[function_name][method]['mean'] = mean_sublinearity
                results[function_name][method]['sem'] = std_error
    
    if results:
        custom_names = ['Standard BO', 'MA Random', 'MA Bandit', 'ABE-BO']
        plot_average_polynomial_order(results, custom_names)
    else:
        print("No results to plot. Check if any valid files were processed.")


if __name__ == "__main__":
    process_files()
    print("Analysis complete. Average polynomial order plot has been generated.")