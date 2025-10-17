import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from transformers.trainer_callback import TrainerCallback


class TrainingCurveCallback(TrainerCallback):
    """Callback to record training metrics for visualization"""

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_accuracies = []
        self.eval_f1s = []
        self.steps = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs and "eval_loss" not in logs:
            self.train_losses.append(logs["loss"])
            self.steps.append(state.global_step)
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            self.eval_accuracies.append(logs.get("eval_accuracy", 0))
            self.eval_f1s.append(logs.get("eval_f1", 0))
            self.eval_steps.append(state.global_step)

    def plot_training_curves(self, strategy: str, save_dir: str = "plots"):
        """Plot training curves and save to file"""
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(12, 5))

        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.steps, self.train_losses, label="Train Loss")
        plt.plot(self.eval_steps, self.eval_losses, label="Validation Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(f"{strategy.upper()} - Loss Curves")
        plt.legend()
        plt.grid(alpha=0.3)

        # Plot metric curves
        plt.subplot(1, 2, 2)
        plt.plot(self.eval_steps, self.eval_accuracies, label="Accuracy")
        plt.plot(self.eval_steps, self.eval_f1s, label="F1 Score")
        plt.xlabel("Steps")
        plt.ylabel("Score")
        plt.title(f"{strategy.upper()} - Evaluation Metrics")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(0.5, 1.0)  # Metrics range for sentiment analysis

        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_curves_{strategy}.png", dpi=300)
        plt.close()
        print(f"Training curves saved to: {save_dir}/training_curves_{strategy}.png")


def plot_confusion_matrix(y_true, y_pred, strategy: str, save_dir: str = "plots"):
    """Plot confusion matrix and save to file"""
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{strategy.upper()} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix_{strategy}.png", dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_dir}/confusion_matrix_{strategy}.png")


def plot_strategy_comparison(full_results, lora_results, save_dir: str = "plots"):
    """Plot comparison between full fine-tuning and LoRA (if both results exist)"""
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data
    strategies = ["Full Fine-tuning", "LoRA Fine-tuning"]
    accuracies = [full_results["accuracy"], lora_results["accuracy"]]
    f1_scores = [full_results["f1"], lora_results["f1"]]
    trainable_params = [
        full_results["trainable_params"] / 1e6,  # Convert to millions
        lora_results["trainable_params"] / 1e6
    ]

    # Create comparison plot
    x = np.arange(len(strategies))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot metrics comparison
    ax1.bar(x - width / 2, accuracies, width, label="Accuracy")
    ax1.bar(x + width / 2, f1_scores, width, label="F1 Score")
    ax1.set_xlabel("Fine-tuning Strategy")
    ax1.set_ylabel("Score")
    ax1.set_title("Performance Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=15)
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")
    ax1.set_ylim(0.8, 1.0)  # Narrow range for better visibility

    # Plot parameter efficiency comparison
    ax2.bar(x, trainable_params, width, color="orange")
    ax2.set_xlabel("Fine-tuning Strategy")
    ax2.set_ylabel("Trainable Parameters (Millions)")
    ax2.set_title("Parameter Efficiency Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=15)
    ax2.grid(alpha=0.3, axis="y")

    # Add value labels on bars
    for i, v in enumerate(trainable_params):
        ax2.text(i, v + 0.1, f"{v:.2f}M", ha="center")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/strategy_comparison.png", dpi=300)
    plt.close()
    print(f"Strategy comparison plot saved to: {save_dir}/strategy_comparison.png")
