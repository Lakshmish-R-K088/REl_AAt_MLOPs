import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os

# Set the style for academic reporting
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def generate_training_plots(csv_file_path, output_filename="training_results_ppo.png"):
    """Reads training data from a CSV and generates performance plots."""
    
    # 1. Data Preparation (Load from CSV)
    if not os.path.exists(csv_file_path):
        print(f"Error: The file '{csv_file_path}' was not found.")
        return

    print(f"Loading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    
    # Verify required columns exist
    required_columns = ['Step', 'Mean_Reward', 'Policy_Loss', 'Entropy', 'Value_Loss']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}' in the CSV file.")
            return

    df['Step_k'] = df['Step'] / 1000 # Convert to kilo-steps for cleaner X-axis

    # 2. Create Figure and Subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Use the filename in the title so you know which run this plot belongs to
    run_name = os.path.basename(csv_file_path).replace('.csv', '')
    fig.suptitle(f'PPO Training Analytics: {run_name}', fontsize=18, fontweight='bold')

    # --- Plot A: Mean Episode Reward ---
    sns.lineplot(ax=axes[0, 0], data=df, x='Step_k', y='Mean_Reward', marker='o', color='#2563eb', linewidth=2.5)
    axes[0, 0].set_title('Mean Episode Reward (Path Efficiency)', fontweight='bold')
    axes[0, 0].set_xlabel('Steps (k)')
    axes[0, 0].set_ylabel('Reward')

    # --- Plot B: Policy Entropy ---
    sns.lineplot(ax=axes[0, 1], data=df, x='Step_k', y='Entropy', marker='s', color='#7c3aed', linewidth=2.5)
    axes[0, 1].set_title('Policy Entropy (Exploration Decay)', fontweight='bold')
    axes[0, 1].set_xlabel('Steps (k)')
    axes[0, 1].set_ylabel('Entropy')

    # --- Plot C: Value Loss (Log Scale) ---
    # Note: Added a small epsilon to Value_Loss to prevent log(0) errors if loss hits exactly 0
    df['Value_Loss_Safe'] = df['Value_Loss'].apply(lambda x: max(x, 1e-10))
    sns.lineplot(ax=axes[1, 0], data=df, x='Step_k', y='Value_Loss_Safe', marker='^', color='#db2777', linewidth=2.5)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Value Loss (Critic Convergence - Log Scale)', fontweight='bold')
    axes[1, 0].set_xlabel('Steps (k)')
    axes[1, 0].set_ylabel('Loss (Log)')

    # --- Plot D: Policy Gradient Loss ---
    sns.lineplot(ax=axes[1, 1], data=df, x='Step_k', y='Policy_Loss', marker='d', color='#16a34a', linewidth=2.5)
    axes[1, 1].set_title('Policy Gradient Loss', fontweight='bold')
    axes[1, 1].set_xlabel('Steps (k)')
    axes[1, 1].set_ylabel('Loss')

    # 3. Final Adjustments
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure dynamically based on the input name
    plt.savefig(output_filename, dpi=300)
    print(f"Graphs generated successfully and saved as '{output_filename}'.")
    
    plt.show()

if __name__ == "__main__":
    # Set up argument parsing so you can pass the file from the terminal
    parser = argparse.ArgumentParser(description="Plot PPO Training Metrics from a CSV file.")
    parser.add_argument("data_file", type=str, help="Path to the CSV file containing the training data.")
    parser.add_argument("--out", type=str, default="training_results.png", help="Name of the output image file.")
    
    args = parser.parse_args()
    
    generate_training_plots(args.data_file, args.out)