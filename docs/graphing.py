import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the CSV files
reward_file_path = "docs/data/10-03-2025_run/10-03-2025_episodes_rewards.csv"
loss_file_path = "docs/data/10-03-2025_run/10-03-2025_episodes_losses.csv"

df_reward = pd.read_csv(reward_file_path)
df_loss = pd.read_csv(loss_file_path)

# Apply a simple moving average for smoothing
window_size = 10  # Adjust this for more or less smoothing
smoothed_rewards = np.convolve(
    df_reward["Value"], np.ones(window_size) / window_size, mode="valid"
)
smoothed_losses = np.convolve(
    df_loss["Value"], np.ones(window_size) / window_size, mode="valid"
)

# Adjust steps to match the smoothed values length
smoothed_reward_steps = df_reward["Step"][: len(smoothed_rewards)]
smoothed_loss_steps = df_loss["Step"][: len(smoothed_losses)]


def plot_graph(df, smoothed_steps, smoothed_values, title, ylabel, color, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")  # Dark gray background for the figure
    ax.set_facecolor("#0d1117")  # Dark gray background for the plot

    # Plot the raw data
    ax.plot(df["Step"], df["Value"], label=title, color=color, alpha=0.8)

    # Plot smoothed line
    ax.plot(
        smoothed_steps,
        smoothed_values,
        label="Smoothed " + ylabel,
        color="red",
        linewidth=2,
        alpha=0.8,
    )

    # Custom grid style for visibility
    ax.grid(True, linestyle="--", alpha=0.5, color="gray")

    # Set tick labels color
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    # Set axis line (spine) colors
    ax.spines["top"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_color("white")

    # Set axis line (spine) visibility
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Labels and title with white text for contrast
    ax.set_xlabel("Episode", color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.set_title(title, color="white")
    # ax.legend()

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# Plot reward graph
plot_graph(
    df_reward,
    smoothed_reward_steps,
    smoothed_rewards,
    "DQN Training Progress",
    "Reward",
    "#1a9aab",
    "docs/media/10-03-2025_run/10-03-2025_episodes_rewards_plot.png",
)

# Plot loss graph
plot_graph(
    df_loss,
    smoothed_loss_steps,
    smoothed_losses,
    "DQN Training Loss",
    "Loss",
    "#ff7f0e",
    "docs/media/10-03-2025_run/10-03-2025_episodes_losses_plot.png",
)
