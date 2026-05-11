import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set the style for academic reporting
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def generate_training_plots():
    # 1. Data Preparation (Extracted from your provided PPO logs)
    data = {
        'Step': [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 40960, 61440, 81920, 100352],
        'Mean_Reward': [-102, -103, -108, -121, -122, -115, -118, -119, -126, -127, -128, -128],
        'Policy_Loss': [3050, 1630, 537, 57.4, 1.9, 0.209, -0.021, -0.021, -0.007, -0.028, 0.013, -0.009],
        'Entropy': [-1.37, -1.22, -0.949, -0.622, -0.538, -0.561, -0.477, -0.458, -0.382, -0.228, -0.160, -0.169],
        'Value_Loss': [7250, 3860, 1440, 244, 8.4, 1.18, 0.00057, 0.00001, 0.000001, 0.000001, 0.000003, 0.000001]
    }
    
    df = pd.DataFrame(data)
    df['Step_k'] = df['Step'] / 1000 # Convert to kilo-steps for cleaner X-axis

    # 2. Create Figure and Subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PPO Training Analytics: Autonomous Infrastructure Drone (policy_v2)', fontsize=18, fontweight='bold')

    # --- Plot A: Mean Episode Reward ---
    sns.lineplot(ax=axes[0, 0], data=df, x='Step_k', y='Mean_Reward', marker='o', color='#2563eb', linewidth=2.5)
    axes[0, 0].set_title('Mean Episode Reward (Path Efficiency)', fontweight='bold')
    axes[0, 0].set_xlabel('Steps (k)')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_ylim(-135, -95)

    # --- Plot B: Policy Entropy ---
    sns.lineplot(ax=axes[0, 1], data=df, x='Step_k', y='Entropy', marker='s', color='#7c3aed', linewidth=2.5)
    axes[0, 1].set_title('Policy Entropy (Exploration Decay)', fontweight='bold')
    axes[0, 1].set_xlabel('Steps (k)')
    axes[0, 1].set_ylabel('Entropy')

    # --- Plot C: Value Loss (Log Scale) ---
    sns.lineplot(ax=axes[1, 0], data=df, x='Step_k', y='Value_Loss', marker='^', color='#db2777', linewidth=2.5)
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
    
    # Save the figure for the report
    plt.savefig('training_results_ppo.png', dpi=300)
    print("Graphs generated successfully and saved as 'training_results_ppo.png'.")
    
    plt.show()

if __name__ == "__main__":
    generate_training_plots()