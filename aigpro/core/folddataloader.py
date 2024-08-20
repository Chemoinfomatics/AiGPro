# ruff: noqa: D102 D103 D107 D101 F841 D105

import os
from typing import Any
from typing import Optional
from typing import Union
import lightning as L

# import pandas as pd
import torch
import torch.multiprocessing
from rich.console import Console
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import Subset
from aigpro.core.pdbdataset import GPCRDataset
from aigpro.core.pdbdataset import GPCRDatasetCombined
from aigpro.data.objects import DM
from aigpro.data.objects import GPCRTargetConfig
from aigpro.utils.utilities import load_dataframe

torch.multiprocessing.set_sharing_strategy("file_system")


console = Console()


def split_or_select(  # noqa: D103
    dataset, split_size=0.9, random_seed=42
):  # -> tuple[Subset[Any], Subset[Any]]:# -> tuple[Subset[Any], Subset[Any]]:
    # all_index = dataset.data.index
    # random.seed(random_seed)

    if isinstance(split_size, float) and 0 < split_size < 1:
        subset1_size: int = int(len(dataset) * split_size)
        subset2_size: int = len(dataset) - subset1_size
        subset1, subset2 = random_split(
            dataset, [subset1_size, subset2_size], generator=torch.Generator().manual_seed(random_seed)
        )
    elif isinstance(split_size, int) and split_size > 1:
        subset1_size = split_size
        subset2_size = len(dataset) - subset1_size
        subset2, subset1 = random_split(
            dataset, [subset1_size, subset2_size], generator=torch.Generator().manual_seed(random_seed)
        )
    else:
        raise ValueError("split_size must be a float between 0 and 1 or an integer greater than 1")

    # return (subset2.indices, subset1.indices)  # (biggenr and lower)
    return Subset(dataset, subset1.indices), Subset(dataset, subset2.indices)


class PDBFoldDataModule(L.LightningDataModule):
    def __init__(  # noqa: D107
        self,
        data: str,
        test: str,
        val: Union[None, str] = None,
        batch_size: int = 32,
        num_workers: Union[None, int] = 10,
        k: Union[int, None] = None,
        num_folds: int = 5,
        split_seed: int = 143,
        pin_memory: bool = False,
        cv: bool = False,
        split_size: Union[None, float] = 0.9,
        dataset_name="GPCR",
        seed=42,
        dataloader=None,
        optimizer_hparams=None,
        learning_rate=0.0001,
        optimizer_name="Adam",
        decay_milestone=None,
        scheduler_name="ReduceLROnPlateau",
        scheduler_monitor="val_loss",
        weight_decay=0.01,
        protein_tokenizer_filename: str = "GPCR_prot_tokenizer.json",
        ligand_tokenizer_filename: str = "GPCR_smi_tokenizer.json",
        msa_col_name: str = "exlcude_terminal",
        y_col_name: str = "pEndPoint",
        state: str = "train",
        dm_filename: str = "GPCR_dm.h5",
        dm_col_iden_name: str = "entry_name",
        test2_files: str = None,
        canonical_smiles_col_name: str = "canonical_smiles",
    ) -> None:
        if num_workers is None:
            num_workers = os.cpu_count()
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.cv: Optional[bool] = cv
        self.seed = seed
        self.dataset_name = dataset_name

        dataload_dict = {
            "GPCR": GPCRDataset,
            "GPCR_combined": GPCRDatasetCombined,
            # "BindingDB": BindingDBDataset,
            # "GPCR_chembl": GPCRChemblDataset,
        }

        self.dataloader = dataload_dict[dataset_name]

    def setup(self, stage=None):  # noqa: D102
        if isinstance(self.hparams.data, str):
            self.hparams.data = load_dataframe(self.hparams.data)

        self.data = self.dataloader(
            self.hparams.data,
            GPCRTargetConfig(
                state=self.hparams.state,
                protein_tokenizer_filename=self.hparams.protein_tokenizer_filename,
                ligand_tokenizer_filename=self.hparams.ligand_tokenizer_filename,
                msa_col_name=self.hparams.msa_col_name,
                y_col_name=self.hparams.y_col_name,
                canonical_smiles_col_name=self.hparams.canonical_smiles_col_name,
                dm=DM(
                    dm_filename=self.hparams.dm_filename,
                    dm_col_iden_name=self.hparams.dm_col_iden_name,
                ),
            ),
        )

        if self.hparams.test is not None:
            if isinstance(self.hparams.test, str):
                self.hparams.test = load_dataframe(self.hparams.test)
            self.data_test = self.dataloader(
                self.hparams.test,
                GPCRTargetConfig(
                    state=self.hparams.state,
                    protein_tokenizer_filename=self.hparams.protein_tokenizer_filename,
                    ligand_tokenizer_filename=self.hparams.ligand_tokenizer_filename,
                    msa_col_name=self.hparams.msa_col_name,
                    y_col_name=self.hparams.y_col_name,
                    dm=DM(
                        dm_filename=self.hparams.dm_filename,
                        dm_col_iden_name=self.hparams.dm_col_iden_name,
                    ),
                ),
            )
        else:
            self.data_test = None

        if self.hparams.test2_files is not None:
            test2_files = self.hparams.test2_files
            if isinstance(self.hparams.test2_files, str):
                test_files = []
                samples = self.hparams.test2_files.split(",")
                for i in samples:
                    i = i.strip()
                    if not os.path.exists(i):
                        # raise FileNotFoundError(f"{i} does not exist")
                        # console.print(f"{i} does not exist")
                        continue
                    test_files.append(i)

                # if "," in self.hparams.test2_files:
                #     test2_files = self.hparams.test2_files.split(",")
                #     test2_files = [x.strip() for x in test2_files]
                # else:
                #     test2_files = [test2_files]
            test2_files = test_files

            assert isinstance(test2_files, list), f"{test2_files} must be a list of file paths"
            self.data_test2 = []
            for test2_file in test2_files:
                if isinstance(test2_file, str):
                    test2_file = load_dataframe(test2_file)
                    print(test2_file.shape)
                self.data_test2.append(
                    self.dataloader(
                        test2_file,
                        GPCRTargetConfig(
                            state=self.hparams.state,
                            protein_tokenizer_filename=self.hparams.protein_tokenizer_filename,
                            ligand_tokenizer_filename=self.hparams.ligand_tokenizer_filename,
                            msa_col_name=self.hparams.msa_col_name,
                            y_col_name=self.hparams.y_col_name,
                            dm=DM(
                                dm_filename=self.hparams.dm_filename,
                                dm_col_iden_name=self.hparams.dm_col_iden_name,
                            ),
                        ),
                    )
                )

        if self.hparams.val is not None:
            if isinstance(self.hparams.val, str):
                self.hparams.val = load_dataframe(self.hparams.val)
            self.data_val = self.dataloader(
                self.hparams.val,
                GPCRTargetConfig(
                    state=self.hparams.state,
                    protein_tokenizer_filename=self.hparams.protein_tokenizer_filename,
                    ligand_tokenizer_filename=self.hparams.ligand_tokenizer_filename,
                    msa_col_name=self.hparams.msa_col_name,
                    y_col_name=self.hparams.y_col_name,
                    dm=DM(
                        dm_filename=self.hparams.dm_filename,
                        dm_col_iden_name=self.hparams.dm_col_iden_name,
                    ),
                ),
            )

        # use same test and val while cv
        if self.cv:
            if self.data_val is None:
                self.data_val = self.data_test
            self.data_train = self.data
        else:
            self.data_train, self.data_val = split_or_select(self.data, self.hparams.split_size, self.seed)

        console.print("#" * 70)
        console.print(f"Total number of samples: {len(self.data)}")
        console.print(f"Total number of training samples: {len(self.data_train)}")
        console.print(f"Total number of validation samples: {len(self.data_val)}")
        console.print(f"Total number of test samples: {len(self.data_test)}")
        console.print("#" * 70)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=False,
            shuffle=False,
            drop_last=True,
            # collate_fn=self.custom_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=False,
            shuffle=False,
            # collate_fn=self.custom_collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any] | list[DataLoader[Any]]:
        original_test = DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=False,
            shuffle=False,
            # collate_fn=self.custom_collate_fn,
        )
        if self.hparams.test2_files is not None:
            test_data_loaders = [
                DataLoader(
                    dataset=self.data_test2[i],
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    persistent_workers=False,
                    shuffle=False,
                    # collate_fn=self.custom_collate_fn,
                )
                for i in range(len(self.data_test2))
            ]
            original_test = [original_test] + test_data_loaders
        # print(original_test, len(original_test))
        return original_test


    def custom_collate_fn(self, batch):
        protein = torch.stack([item[0] for item in batch[0]])
        ligand = torch.stack([item[1] for item in batch[0]])
        smile_desc = torch.stack([item[2] for item in batch[0]])
        charge_fp = torch.stack([item[3] for item in batch[0]])
        y = torch.stack([item[4] for item in batch[1]])
        return protein, ligand, smile_desc, charge_fp, y

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            shuffle=False,
            # collate_fn=self.custom_collate_fn,
        )
