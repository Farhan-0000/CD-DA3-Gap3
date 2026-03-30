import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
import os

def visualize_results():
    if not os.path.exists(config.METRICS_RESULTS_FILE):
        print("No metrics file found to plot. Run main.py first.")
        return
        
    df = pd.read_csv(config.METRICS_RESULTS_FILE)
    
    # Plot 1: Defense Effectiveness (Harmful Rate vs Budget)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="budget", y="harmful", hue="explainer", marker="o")
    plt.title("Tokens Excision vs. Harmful Response Rate")
    plt.xlabel("Deletion Budget (Ratio of Tokens)")
    plt.ylabel("Harmful Response Rate")
    plt.grid(True)
    plt.savefig(f"{config.RESULTS_DIR}/harmful_rate_vs_budget.png")
    
    # Plot 2: Adversarial Token Recovery
    # Filter out random if we only care about Shap vs Attention, or keep it to show Random's linear recovery
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="budget", y="recovery_rate", hue="explainer", marker="o")
    plt.title("Adversarial Token Recovery (SHAP vs Attention vs Random)")
    plt.xlabel("Budget (Top p% of Tokens)")
    plt.ylabel("Fraction of Adversarial Suffix Tokens Recovered")
    plt.grid(True)
    plt.savefig(f"{config.RESULTS_DIR}/recovery_rate_vs_budget.png")
    
    print("Visualizations saved to results directory.")

if __name__ == "__main__":
    visualize_results()
