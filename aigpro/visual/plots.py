import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from sklearn.metrics import r2_score

console = Console()


def print_table(sample) -> None:
    """Prints a table of the results.

    Args:
        sample (_type_): _description_
    """
    table: Table = Table(title="CV Results in Detail")
    table.add_column("Metric/Model", style="cyan", no_wrap=True)
    max_len: int = max(len(x) for x in sample.values())
    for x in range(max_len):
        table.add_column(f"Fold {str(x + 1)}", justify="right", style="magenta", no_wrap=True)
    table.add_column("Mean", justify="right", style="green")
    # mean add column at last line
    for key, value in sample.items():
        table.add_row(key, *[str(x) for x in value] + [str(sum(value) / len(value))])
    console.print(table)


def scatter_plot(y_pred, y_true, color="black", title="True vs Predicted", linewidth=2):
    """Scatter plot of predicted vs true values.

    Args:
        y_pred (_type_): _description_
        y_true (_type_): _description_
        color (str, optional): _description_. Defaults to "black".
        title (str, optional): _description_. Defaults to "True vs Predicted".
        linewidth (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), dpi=300, sharey=False, sharex=False)
    sns.scatterplot(x=y_pred, y=y_true, ax=ax[0], color=color, s=20, facecolor="maroon", alpha=0.2)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    ax[0].set_title(f"{title} (R2: {r2_score(y_true, y_pred):.3f}): {len(y_pred)}")
    ax[0].plot([min_val, max_val], [min_val, max_val], "--", lw=linewidth, alpha=0.7, color=color)
    ax[0].set_xlim([y_true.min(), y_true.max()])
    ax[0].set_ylim([y_true.min(), y_true.max()])
    sns.kdeplot(x=y_pred, y=y_true, ax=ax[1], fill=True, cmap="Reds", warn_singular=False)
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")
    ax[1].set_title(f"{title} (R2: {r2_score(y_true, y_pred):.3f}): {len(y_pred)}")
    ax[1].plot([min_val, max_val], [min_val, max_val], "--", lw=linewidth, alpha=0.7, color=color)
    ax[1].set_xlim([y_true.min(), y_true.max()])
    ax[1].set_ylim([y_true.min(), y_true.max()])
    ax[1].set_aspect("equal")
    plt.close()
    return fig
