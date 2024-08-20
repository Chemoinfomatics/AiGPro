from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from lifelines.utils import concordance_index
from rich.console import Console
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

console = Console()


def sensivity_specifity_cutoff(y_true, y_score):
    """Find data-driven cut-off for classification.

    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).

    References
    ----------
    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.

    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.

    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either-or presence-absence. Acta oecologica, 31(3), 361-369.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]


def q2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """_summary_.

    Args:
        y_true (np.ndarray): _description_
        y_pred (np.ndarray): _description_

    Returns:
        float: _description_
    """
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    _q2 = 1 - (rss / tss)
    return _q2


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | Any:
    """Return the coefficient of determination.

    Args:
        y_true (np.ndarray): _description_
        y_pred (np.ndarray): _description_

    Returns:
        float: _description_
    """
    return r2_score(y_true, y_pred)


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the Pearson correlation coefficient.

    Args:
        y_true (np.ndarray): _description_
        y_pred (np.ndarray): _description_

    Returns:
        float: _description_
    """
    pearson_res, _ = stats.pearsonr(y_true, y_pred)
    return pearson_res


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the Spearman correlation coefficient.

    Args:
        y_true (np.ndarray): _description_
        y_pred (np.ndarray): _description_

    Returns:
        float: _description_
    """
    spearman_res, _ = stats.spearmanr(y_true, y_pred)
    return spearman_res


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean absolute error.

    Args:
        y_true (np.ndarray): _description_
        y_pred (np.ndarray): _description_

    Returns:
        float: _description_
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float | np.ndarray | Any:
    """_summary_.

    Args:
        y_true (np.ndarray): _description_
        y_pred (np.ndarray): _description_

    Returns:
        float: _description_
    """
    return mean_squared_error(y_true, y_pred, squared=False)


def q2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the Q2 score.

    Args:
        y_true (np.ndarray): _description_
        y_pred (np.ndarray): _description_

    Returns:
        float: _description_
    """
    return q2_score(y_true, y_pred)


def r2m(y_true, y_pred):
    """Return the regression toward the mean.

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    r2 = np.corrcoef(y_true, y_pred, rowvar=True)[0][1] ** 2
    r20 = np.corrcoef(y_true, y_pred, rowvar=False)[0][1] ** 2
    r2m = 1 - np.sqrt((r2 - r20))
    return r2 * r2m


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float | Any:
    """Return the mean squared error.

    Args:
        y_true (np.ndarray): _description_
        y_pred (np.ndarray): _description_

    Returns:
        float: _description_
    """
    return mean_squared_error(y_true, y_pred, squared=True)


def metric(y_true: np.ndarray, y_pred: np.ndarray, verbose=True) -> dict:
    """Calculate and print the metrics.

    Args:
        y_true (np.ndarray): y_true
        y_pred (np.ndarray): y_pred
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        dict: Metric dictionary
    """
    metric_dict = {}

    metric_dict["R2"] = r2(y_true, y_pred)
    metric_dict["R2M"] = r2m(y_true, y_pred)
    metric_dict["MSE"] = mse(y_true, y_pred)
    metric_dict["R"] = pearson(y_true, y_pred)
    metric_dict["Spearman"] = spearman(y_true, y_pred)
    metric_dict["MAE"] = mae(y_true, y_pred)
    metric_dict["RMSE"] = rmse(y_true, y_pred)
    metric_dict["Q2"] = q2(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)
    metric_dict["CI"] = ci
    if verbose:
        console.rule("[bold red]Metric Results")

        for metric_name, metric_value in metric_dict.items():
            console.print(f"{metric_name}: {metric_value:.3f}")

        console.rule("[bold red]END")
    return metric_dict


def metric_old(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Calculate and print the metrics.

    Args:
        y_true (np.ndarray): y_true
        y_pred (np.ndarray): y_pred
    """
    console.rule("[bold red]Metric Results")
    console.print(
        f"R2: {r2(y_true, y_pred):.3f}",
    )
    console.print(f"Regression toward mean: {r2m(y_true, y_pred):.3f}")
    console.print(f"MSE: {mse(y_true ,y_pred):.3f}")
    console.print(f"Pearson: {pearson(y_true, y_pred):.3f}")
    console.print(f"Spearman: {spearman(y_true, y_pred):.3f}")
    console.print(f"MAE: {mae(y_true, y_pred):.3f}")
    console.print(f"RMSE: {rmse(y_true, y_pred):.3f}")
    console.print(f"Q2: {q2(y_true, y_pred):.3f}")
    ci = concordance_index(y_true, y_pred)
    console.print(f"Concordance Index: {ci:.3f}")
    console.rule("[bold red]END")


def convert_IC50_to_pIC50_or_reverse(value: float, conversion_type: str, unit: str = "nM") -> float:
    """Convert IC50 to pIC50 or pIC50 to IC50.

    Args:
        value (float): _description_
        conversion_type (str): _description_
        unit (str, optional): _description_. Defaults to "nM".

    Raises:
        ValueError: _description_

    Returns:
        float: _description_
    """  # Determine the metric conversion factor for the specified unit
    if conversion_type == "IC50_to_pIC50":
        # Convert IC50 to pIC50 using the formula pIC50 = -log(IC50) + log(metric_convert)
        return ic50_to_pic50(value, unit=unit)
    elif conversion_type == "pIC50_to_IC50":
        # Convert pIC50 to IC50 using the formula IC50 = 10^(-pIC50) / metric_convert
        return pic50_to_ic50(value, unit=unit)
    else:
        raise ValueError(f"Invalid conversion type: {conversion_type}")


def ic50_to_pic50(ic50_value: float, unit: str = "nM") -> float:
    """Convert IC50 value in nM to pIC50 value."""
    # Convert IC50 value in nM to pIC50 value.
    # Based on https://en.wikipedia.org/wiki/IC50
    # IC50 is the concentration of an inhibitor that inhibits 50% of the activity of a drug.
    # REF https://www.biorxiv.org/content/biorxiv/early/2022/10/18/2022.10.15.512366.full.pdf
    # Calculate pIC50

    pic50 = _get_metric_convert(unit) - np.log10(ic50_value)
    # Return pIC50
    return pic50


def pic50_to_ic50(pic50_value: float, unit: str = "nM") -> float:
    """Convert IC50 value in nM to pIC50 value."""
    # Convert IC50 value in nM to pIC50 value.
    # Based on https://en.wikipedia.org/wiki/IC50
    # IC50 is the concentration of an inhibitor that inhibits 50% of the activity of a drug.
    # REF https://www.biorxiv.org/content/biorxiv/early/2022/10/18/2022.10.15.512366.full.pdf
    # Calculate pIC50

    metric_convert = _get_metric_convert(unit)
    IC50 = pow(10, (metric_convert - pic50_value))
    return IC50


def _get_metric_convert(unit: str) -> float:
    """Return the correct metric conversion based on the unit."""
    # Use the correct metric conversion based on the unit
    if unit == "nM":
        metric_convert = 9
    elif unit == "M":
        metric_convert = 1
    elif unit == "uM":
        metric_convert = 6
    elif unit == "mM":
        metric_convert = 3
    elif unit == "pM":
        metric_convert = 12
    else:
        raise ValueError(f"Unit not supported: {unit}")

    return metric_convert


def binary_classification_metrics(y_true, y_pred, plot=True):
    """Generate binary classification metrics.

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        plot (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    mcc = matthews_corrcoef(y_true, y_pred)
    positive_rate = np.array(y_true).sum() / len(y_true)
    negative_rate = 1 - positive_rate
    BA = (TP / (TP + FN) + TN / (TN + FP)) / 2

    total = len(y_true)
    print(f"Total: {total}")
    print(f"Positive Ratio: {positive_rate:.3f}")
    print(f"Negative Ratio: {negative_rate:.3f}")
    print(f"MCC: {mcc:.3f}")
    print(f"BA: {BA:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 score: {f1:.3f}")
    print(f"ROC AUC score: {roc_auc:.3f}")
    # print("Confusion Matrix:")
    # print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    if plot:
        # Generate confusion matrix plot
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            title="Confusion Matrix",
            ylabel="True label",
            xlabel="Predicted label",
        )

        # Add annotations to the confusion matrix plot
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        plt.show()

        # Generate ROC curve plot
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()

    return dict(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        tn=TN,
        fp=FP,
        fn=FN,
        tp=TP,
        mcc=mcc,
        positive_rate=positive_rate,
        negative_rate=negative_rate,
        total=total,
        BA=BA,
    )
