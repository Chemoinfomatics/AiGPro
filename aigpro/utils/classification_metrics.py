import matplotlib.pyplot as plt
import seaborn as sns
import torchmetrics
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from torchmetrics.functional.classification import binary_f1_score


def evaluate_classification(y_pred, y_true, threshold=0.5):  # noqa: D417
    """Evaluate classification metrics and generate relevant plots.

    Parameters:
    - y_true: True labels (ground truth).
    - y_pred_prob: Predicted probabilities for the positive class.
    - threshold: Classification threshold (default is 0.5).

    Returns:
    - a dictionary containing the following metrics:
        - confusion_matrix
        - accuracy
        - precision
        - recall
        - f1
        - roc_auc
        - pr_auc
    and plots for ROC and PR curves.
    """

    confusion_matrix = torchmetrics.functional.confusion_matrix(
        y_pred, y_true, task="binary", num_classes=2, num_labels=2
    )
    accuracy = torchmetrics.functional.accuracy(y_pred, y_true, task="binary")
    precision = torchmetrics.functional.precision(y_pred, y_true, task="binary")
    recall = torchmetrics.functional.recall(y_pred, y_true, task="binary")
    f1 = binary_f1_score(
        y_pred,
        y_true,
    )
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot ROC Curve
    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve")
    axes[0].legend()
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    axes[1].plot(recall, precision, color="green", lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    classes = ["Agonist", "Antagonist"]

    sns.heatmap(confusion_matrix.cpu().numpy(), annot=True, ax=axes[2], fmt="d")
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xlabel("Predicted Label")
    axes[2].set_ylabel("True Label")
    axes[2].set_xticks([0.5, 1.5])
    axes[2].set_yticks([0.5, 1.5])
    axes[2].set_xticklabels(classes)
    axes[2].set_yticklabels(classes)

    plt.tight_layout()

    return {
        "confusion_matrix": confusion_matrix,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }, fig
