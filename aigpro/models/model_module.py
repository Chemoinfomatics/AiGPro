# ruff: noqa: D102 D103 D107 D101 F841

import math
import os
from dataclasses import asdict
from functools import lru_cache
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
import lightning as L
import torch
from lifelines.utils import concordance_index

# from lightning.pytorch.cli import MODEL_RESGISTRY
from rich.console import Console

# from seqvec.analysis.metrics import concordance_index_compute
from torch import Tensor
from torchmetrics import MatthewsCorrCoef
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics import PearsonCorrCoef
from torchmetrics import R2Score
from torchmetrics import SpearmanCorrCoef
from torchmetrics.regression import MeanAbsolutePercentageError

# from torchmetrics.classification import BinaryF1Score
import wandb
from gpcrscan.data.objects import MetricData
from gpcrscan.data.objects import Metrics
from gpcrscan.models.model import TransKinase_new
from gpcrscan.utils.classification_metrics import evaluate_classification
from gpcrscan.utils.logger import get_logger
from gpcrscan.utils.utilities import print_metric_table
from gpcrscan.visual.plots import scatter_plot

console = Console()
logger = get_logger()
# logger.setLevel("INFO")


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3) -> Callable[..., float]:
    def scaler(x):
        return 1.0

    def lr_lambda(it):
        return min_lr + (max_lr - min_lr) * relative(it, stepsize)

    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


# @MODEL_RESGISTRY
class PDBModelModule_old(L.LightningModule):
    def __init__(
        self,
        model=None,
        learning_rate: float = 1e-3,
        optimizer_name: str = "Adam",
        batch_size: int = 32,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        num_workers: Union[int, None] = None,
        weight_decay=1e-2,
        scheduler_name: str = "ReduceLROnPlateau",
        scheduler_monitor: str = "loss",
        decay_milestone: Union[None, List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # self.save_hyperparameters(logger=False)
        decay_milestone = decay_milestone or [50, 80, 110, 150, 200, 220, 250]
        # self.lr: float = lr
        self.scheduler_monitor: str = scheduler_monitor
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.num_workers: Union[int, None] = num_workers or os.cpu_count()
        self.model = model if model is not None else TransKinase_new
        # self.loss_module = nn.MSELoss()
        self.mse: MeanSquaredError = MeanSquaredError()
        self.rmse: MeanSquaredError = MeanSquaredError(squared=False)
        self.r2: R2Score = R2Score()
        self.mae: MeanAbsoluteError = MeanAbsoluteError()
        self.spearmanr: SpearmanCorrCoef = SpearmanCorrCoef()
        self.pearsonr = PearsonCorrCoef()
        self.smooth_l1_loss = torch.nn.SmoothL1Loss()
        # self.ev: ExplainedVariance = ExplainedVariance()
        # self.CustomCI = concordance_index_compute
        self.result_dict_train = {}
        self.result_dict_test = {}
        self.result_dict_valid = {}
        self.schedulers_name = scheduler_name
        self.alpha_custom = 0.7
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters(ignore=["model"])
        self.automatic_optimization = True  # than manualy

    def configure_optimizers(self) -> Tuple[List, List]:
        #  support Adam or SGD as optimizers.
        if self.hparams.optimizer_name in ["Adam", "AdamW"]:
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
            )
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
            )
        else:
            assert False, f'Unknown optimizer: "{self.optimizer_name}"'

        if self.schedulers_name == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.decay_milestone, gamma=0.5)
        elif self.schedulers_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)
        elif self.schedulers_name == "ReduceLROnPlateau":
            # decay lr if no improvement in loss
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.8,
                patience=3,
                verbose=True,
                threshold=0.0001,
                threshold_mode="rel",
                cooldown=0,
                min_lr=0,
                eps=1e-08,
            )
        elif self.schedulers_name == "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss",
                "interval": "epoch",
                "frequency": 5,
            },
        }
        # return [optimizer], []

    @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.accumulate_grad_batches * self.epochs

    def forward(self, x) -> Tensor:  # noqa: D102
        return self.model(x)

    def training_step(self, batch, batch_idx) -> dict:  # noqa: D102
        # opt = self.optimizers()
        x, y_true = batch
        y_pred = self.all_prediction(x)
        loss = self.compute_loss(y_pred, y_true)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        tensorboard_logs: dict[str, Tensor] = {"train_rmse_loss": loss}
        progress_bar_metrics: dict[str, Tensor] = tensorboard_logs
        return {
            "loss": loss,
            "pred": y_pred,
            "true": y_true,
        }

    def compute_loss(self, y_pred, y_true):
        loss = self.rmse(y_pred, y_true)
        return loss  # * 100  # scaling loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y_true = batch
        # x = x.view(x.size(0), -1)
        y_pred = self.all_prediction(x)
        loss = self.compute_loss(y_pred, y_true)
        self.validation_step_outputs.append(
            {
                "val_loss": loss,
                "pred": y_pred,
                "true": y_true,
            }
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {
            "val_loss": loss,
            "pred": y_pred,
            "true": y_true,
        }
        # return [y_pred, y_true]

    def on_validation_epoch_end(self) -> None:  # noqa: D102
        _y_pred, _y_true, results = self.metric_and_log(self.validation_step_outputs, title="val", log_plot=True)
        self.validation_step_outputs.clear()
        print_metric_table("Validation Metrics", asdict(results))
        del results

    def all_prediction(self, x):  # noqa: D102
        result = self.model(x)
        return result.flatten()

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y_true = batch
        y_pred = self.model(x)
        y_pred: torch.Tensor = torch.flatten(y_pred)
        loss = self.compute_loss(y_pred, y_true)
        self.test_step_outputs.append(
            {
                "val_loss": loss,
                "pred": y_pred,
                "true": y_true,
            }
        )
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {
            "loss": loss,
            "pred": y_pred,
            "true": y_true,
        }

    def on_test_epoch_end(self) -> None:  # v2.0.0>
        _y_pred, _y_true, results = self.metric_and_log(self.test_step_outputs, title="test", log_plot=True)
        print_metric_table("Test Metrics", asdict(results))
        self.test_step_outputs.clear()

    def get_backbone(self):
        return self.model

    def compute_metrics(self, y_pred, y_true):
        y_pred = y_pred.to(torch.float32)
        y_true = y_true.to(torch.float32)
        metric_mse = self.mse(y_pred, y_true)
        metric_mae = self.mae(y_pred, y_true)
        metric_rmse = self.rmse(y_pred, y_true)
        metric_r2 = self.r2(y_pred, y_true)
        metric_spear = self.spearmanr(y_pred, y_true)
        metric_pearsonr = self.pearsonr(y_pred, y_true)
        # concordnace_index = concordance_index_compute(y_pred, y_true)
        concordnace_index = concordance_index(y_true.cpu().numpy(), y_pred.cpu().numpy())
        return (
            metric_mse,
            metric_mae,
            metric_rmse,
            metric_r2,
            metric_spear,
            metric_pearsonr,
            concordnace_index,
        )

    def metric_and_log(self, outputs, title, log_plot=True):
        y_pred, y_true = zip(*[(x["pred"], x["true"]) for x in outputs if all(k in x for k in ("pred", "true"))])
        y_pred = torch.cat(y_pred, dim=0).detach()
        y_true = torch.cat(y_true, dim=0).detach()
        (
            _mse,
            _mae,
            _rmse,
            _r2,
            _spear,
            _pearsonr,
            _ci,
        ) = self.compute_metrics(y_pred, y_true)
        results = Metrics(
            mse=_mse,
            rmse=_rmse,
            mae=_mae,
            r2=_r2,
            spearman=_spear,
            pearson=_pearsonr,
            ci=_ci,
        )
        self.log_dict(
            {
                f"{title}_mse": results.mse,
                f"{title}_rmse": results.rmse,
                f"{title}_r2": results.r2,
                f"{title}_spear": results.spearman,
                f"{title}_ci": results.ci,
                f"{title}_mae": results.mae,
                f"{title}_pearsonr": results.pearson,
            },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        wandb.log(
            {
                f"{title}_mse": results.mse,
                f"{title}_rmse": results.rmse,
                f"{title}_r2": results.r2,
                f"{title}_spear": results.spearman,
                f"{title}_ci": results.ci,
                f"{title}_mae": results.mae,
                f"{title}_pearsonr": results.pearson,
            }
        )
        if log_plot:
            fig = scatter_plot(
                y_pred.cpu().numpy(),
                y_true.cpu().numpy(),
                title=f"{title} scatter plot",
            )
            wandb.log({f"{title}_scatter_plot": wandb.Image(fig)})
            fig.clf()
        return y_pred, y_true, results

    def log_classification(self, y_pred_class, y_label):
        classification_results, class_figure = evaluate_classification(y_pred_class, y_label)
        self.log_dict(
            {
                "accuracy": classification_results["accuracy"],
                "precision": classification_results["precision"],
                "recall": classification_results["recall"],
                "f1": classification_results["f1"],
                "roc_auc": classification_results["roc_auc"],
                "pr_auc": classification_results["pr_auc"],
            },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        wandb.log(
            {
                "accuracy": classification_results["accuracy"],
                "precision": classification_results["precision"],
                "recall": classification_results["recall"],
                "f1": classification_results["f1"],
                "roc_auc": classification_results["roc_auc"],
                "pr_auc": classification_results["pr_auc"],
            }
        )
        wandb.log({"classification_plot": wandb.Image(class_figure)})
        return classification_results, class_figure


# @MODEL_RESGISTRY
class PDBModelModule(L.LightningModule):
    def __init__(
        self,
        model=None,
        learning_rate: float = 1e-4,
        optimizer_name: str = "Adam",
        batch_size: int = 32,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        num_workers: Union[int, None] = None,
        weight_decay=1e-2,
        scheduler_name: str = "ReduceLROnPlateau",
        scheduler_monitor: str = "loss",
        decay_milestone: Union[None, List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # self.save_hyperparameters(logger=False)
        decay_milestone = decay_milestone or [50, 80, 110, 150, 200, 220, 250]
        # self.lr: float = lr
        self.multi = False
        self.scheduler_monitor: str = scheduler_monitor
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.num_workers: Union[int, None] = num_workers or os.cpu_count()
        self.model = model
        # self.loss_module = nn.MSELoss()
        self.mse: MeanSquaredError = MeanSquaredError()
        self.rmse: MeanSquaredError = MeanSquaredError(squared=False)
        self.r2: R2Score = R2Score()
        self.mae: MeanAbsoluteError = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()

        self.spearmanr: SpearmanCorrCoef = SpearmanCorrCoef()
        self.pearsonr = PearsonCorrCoef()
        self.matthews_corrcoef = MatthewsCorrCoef(task="binary")
        self.smooth_l1_loss = torch.nn.SmoothL1Loss()
        # self.ev: ExplainedVariance = ExplainedVariance()
        # self.CustomCI = concordance_index_compute
        self.result_dict_train = {}
        self.result_dict_test = {}
        self.result_dict_valid = {}
        self.schedulers_name = scheduler_name
        self.alpha_custom = 0.7
        self.validation_step_outputs = []
        self.test_step_outputs = {}
        self.save_hyperparameters(ignore=["model"])
        self.automatic_optimization = True  # than manualy

    def configure_optimizers(self) -> Tuple[List, List]:
        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
            )
        elif self.hparams.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
            )
        elif self.hparams.optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.learning_rate,
            )
        elif self.hparams.optimizer_name == "Adadelta":
            optimizer = torch.optim.Adadelta(
                self.parameters(),
                lr=self.learning_rate,
            )
        elif self.hparams.optimizer_name == "Adagrad":
            optimizer = torch.optim.Adagrad(
                self.parameters(),
                lr=self.learning_rate,
            )
        else:
            assert False, f'Unknown optimizer: "{self.optimizer_name}"'

        if self.hparams.scheduler_name == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.decay_milestone, gamma=0.5)
        elif self.hparams.scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)
        elif self.hparams.scheduler_name == "ReduceLROnPlateau":
            # decay lr if no improvement in loss
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.8,
                patience=3,
                verbose=True,
                threshold=0.0001,
                threshold_mode="rel",
                cooldown=0,
                min_lr=1e-12,
                eps=1e-08,
            )
        elif self.hparams.scheduler_name == "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss",
                "interval": "epoch",
                "frequency": 5,
            },
        }
        # return [optimizer], []

    @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.accumulate_grad_batches * self.epochs

    def forward(self, x) -> Tensor:  # noqa: D102
        return self.model(x)

    def training_step(self, batch, batch_idx) -> dict:  # noqa: D102
        loss, y_true, y_pred = self.compute_loss(batch)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        tensorboard_logs: dict[str, Tensor] = {"train_rmse_loss": loss}
        progress_bar_metrics: dict[str, Tensor] = tensorboard_logs
        return {
            "loss": loss,
            "pred": y_pred,
            "true": y_true,
        }

    def compute_loss(self, batch):
        x, y_true = batch
        y_true, y_label = y_true
        y_pred_class, y_pred = self.all_prediction(x)
        loss = self.mse(y_pred, y_true)
        return loss, y_true, y_pred  # * 100  # scaling loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        loss, y_true, y_pred = self.compute_loss(batch)
        self.validation_step_outputs.append(
            {
                "val_loss": loss,
                "pred": y_pred,
                "true": y_true,
            }
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {
            "val_loss": loss,
            "pred": y_pred,
            "true": y_true,
        }

    def on_validation_epoch_end(self) -> None:  # noqa: D102
        _y_pred, _y_true, results = self.metric_and_log(self.validation_step_outputs, title="val", log_plot=True)
        print_metric_table("Validation Metrics", asdict(results))
        self.validation_step_outputs.clear()
        del results

    def all_prediction(self, x):  # noqa: D102
        result = self.model(x)
        result = result.flatten()
        return result
        result_class = result[:, :2]
        result_class = torch.argmax(result_class, dim=1)
        result_reg = result[:, 2:]
        return result_class, result_reg

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, dataloader_idx) -> Dict[str, Tensor]:  # noqa: D102
        if dataloader_idx > 0:
            self.multi = True
        loss, y_true, y_pred = self.compute_loss(batch)
        if dataloader_idx not in self.test_step_outputs:
            self.test_step_outputs[dataloader_idx] = {
                "val_loss": [],
                "pred": [],
                "true": [],
            }
        self.test_step_outputs[dataloader_idx]["val_loss"].append(loss)
        self.test_step_outputs[dataloader_idx]["pred"].append(y_pred)
        self.test_step_outputs[dataloader_idx]["true"].append(y_true)
        self.log(f"test_loss_{dataloader_idx}", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {
            f"loss_{dataloader_idx }": loss,
            f"pred_{dataloader_idx }": y_pred,
            f"true_{dataloader_idx }": y_true,
        }

    def on_test_epoch_end(self) -> None:  # v2.0.0>
        if self.multi:
            _y_pred, _y_true, results = self.multi_compute_metrics(self.test_step_outputs)
            for i in range(len(results)):
                print_metric_table(f"Test Metrics {i}", asdict(results[i]))
        else:
            _y_pred, _y_true, results = self.metric_and_log(self.test_step_outputs, title="test", log_plot=True)
            print_metric_table("Test Metrics", asdict(results))
        self.test_step_outputs.clear()

    def get_backbone(self):
        return self.model

    def compute_metrics(self, y_pred, y_true):
        y_pred = y_pred.to(torch.float32)
        y_true = y_true.to(torch.float32)
        metric_mse = self.mse(y_pred, y_true)
        metric_mae = self.mae(y_pred, y_true)
        metric_rmse = self.rmse(y_pred, y_true)
        metric_r2 = self.r2(y_pred, y_true)
        metric_spear = self.spearmanr(y_pred, y_true)
        metric_pearsonr = self.pearsonr(y_pred, y_true)
        # concordnace_index = concordance_index_compute(y_pred, y_true)
        concordnace_index = concordance_index(y_true.cpu().numpy(), y_pred.cpu().numpy())
        return (
            metric_mse,
            metric_mae,
            metric_rmse,
            metric_r2,
            metric_spear,
            metric_pearsonr,
            concordnace_index,
        )

    def multi_compute_metrics(self, test_step_outputs: dict):
        all_y_pred, y_true, results = [], [], []
        for k, v in test_step_outputs.items():
            outputs = []
            y_pred = []
            y_true = []
            y_pred.append(v["pred"])
            y_true.append(v["true"])
            for i in range(len(v["pred"])):
                outputs.append({"pred": v["pred"][i], "true": v["true"][i]})
            _y, _t, _r = self.metric_and_log(outputs, title=f"test_{k}", log_plot=True)
            all_y_pred.append(_y)
            y_true.append(_t)
            results.append(_r)

        return all_y_pred, y_true, results

    def metric_and_log(self, outputs, title, log_plot=True):
        y_pred, y_true = zip(*[(x["pred"], x["true"]) for x in outputs if all(k in x for k in ("pred", "true"))])

        y_pred = torch.cat(y_pred, dim=0).detach()
        y_true = torch.cat(y_true, dim=0).detach()
        (
            _mse,
            _mae,
            _rmse,
            _r2,
            _spear,
            _pearsonr,
            _ci,
        ) = self.compute_metrics(y_pred, y_true)
        results = Metrics(
            mse=_mse,
            rmse=_rmse,
            mae=_mae,
            r2=_r2,
            spearman=_spear,
            pearson=_pearsonr,
            ci=_ci,
        )
        self.log_dict(
            {
                f"{title}_mse": results.mse,
                f"{title}_rmse": results.rmse,
                f"{title}_r2": results.r2,
                f"{title}_spear": results.spearman,
                f"{title}_ci": results.ci,
                f"{title}_mae": results.mae,
                f"{title}_pearsonr": results.pearson,
            },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        wandb.log(
            {
                f"{title}_mse": results.mse,
                f"{title}_rmse": results.rmse,
                f"{title}_r2": results.r2,
                f"{title}_spear": results.spearman,
                f"{title}_ci": results.ci,
                f"{title}_mae": results.mae,
                f"{title}_pearsonr": results.pearson,
            }
        )
        if log_plot:
            fig = scatter_plot(
                y_pred.cpu().numpy(),
                y_true.cpu().numpy(),
                title=f"{title} scatter plot",
            )
            wandb.log({f"{title}_scatter_plot": wandb.Image(fig)})
            fig.clf()
        return y_pred, y_true, results


# @MODEL_RESGISTRY
class PDBModelModuleGPCR(PDBModelModule):
    def compute_loss(self, batch, log=False) -> tuple[Any, Any, Any, MetricData] | tuple[Any, Any, Any]:
        x, y = batch
        y_true, y_label = y
        y_label = y_label.to(torch.long)
        y_pred_class, y_pred = self.all_prediction(x)
        # y_pred= self.all_prediction(x)
        y_pred_class = y_label
        logger.info(f"y_pred_class: {y_pred_class}")
        logger.info(f"y_true: {y_true}")
        logger.info(f"y_label: {y_label}")
        logger.info(f"y_pred: {y_pred}")
        reg_loss = self.rmse(y_pred, y_true)  # * 0.5 + (1 - r2) * 0.5
        alpha = 0.5  # Weight for classification loss
        beta = 1 - alpha  # Weight for regression loss
        epsilon = 1e-6
        # loss = (alpha * class_loss) + (beta * reg_loss) + epsilon
        loss = reg_loss + epsilon  # + mape
        logger.info(f"reg_loss: {reg_loss}")
        if log:
            data = MetricData(
                loss=loss,
                y_pred=y_pred,
                y_true=y_true,
                y_pred_class=y_pred_class,
                y_label=y_label,
            )
            return loss, y_true, y_pred, data
        return loss, y_true, y_pred  # * 100  # scaling loss

    def all_prediction(self, x) -> tuple[Any, Any]:  # noqa: D102
        result = self.model(x)
        result = result.flatten()
        result_reg = result
        result_class = torch.sigmoid(result_reg)
        logger.info(f"result_reg: {result_reg}")
        return result_class, result_reg

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        loss, y_true, y_pred, data = self.compute_loss(batch, log=True)  # type: ignore

        self.validation_step_outputs.append(
            {
                "val_loss": loss,
                "pred": y_pred,
                "true": y_true,
                "y_pred_class": data.y_pred_class,
                "y_label": data.y_label,
            }
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {
            "val_loss": loss,
            "pred": y_pred,
            "true": y_true,
            "y_pred_class": data.y_pred_class,
            "y_label": data.y_label,
        }  # type: ignore
        # return [y_pred, y_true]

    def on_validation_epoch_end(self) -> None:  # noqa: D102
        _y_pred, _y_true, results = self.metric_and_log(self.validation_step_outputs, title="val", log_plot=True)
        self.validation_step_outputs.clear()
        print_metric_table("Validation Metrics", asdict(results))
        del results

    @torch.jit.export
    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int | None = None) -> Tensor:
        x = batch
        y_pred_class, y_pred = self.all_prediction(x)
        return y_pred
