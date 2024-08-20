# ruff: noqa: D102 D103 D107 D101 F841 D417

from ast import literal_eval
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from torch import Tensor
from aigpro.chem.descriptors import morgan_fingerprint
from aigpro.chem.old_tokenizer import get_pad_array
from aigpro.chem.old_tokenizer import tokenize_smiles
from aigpro.chem.tokenizers import load_custom_tokenizer
from aigpro.data.objects import GPCRTargetConfig
from aigpro.pretrained.embeddings import GetEmbeddings
from aigpro.utils import logger
from aigpro.utils.utilities import load_dataframe

console = Console()
log = logger.get_logger()
# log.setLevel("INFO")



class GPCRDatasetCombined(torch.utils.data.Dataset):  # noqa: D101
    def __init__(
        self,
        data,
        gpcr: GPCRTargetConfig | None = None,
    ) -> None:
        if not isinstance(gpcr, GPCRTargetConfig):
            raise TypeError("gpcr_target_config must be of type GPCRTargetConfig")
        self.gpcr: GPCRTargetConfig = gpcr
        if isinstance(data, str):
            dataframe = load_dataframe(data)
        else:
            dataframe = data
        self.data: pd.DataFrame = dataframe
        self.ProteinTokenizer = load_custom_tokenizer(
            self.gpcr.protein_tokenizer_filename,
        )
        self.SmiTokenizer = load_custom_tokenizer(
            self.gpcr.ligand_tokenizer_filename,
        )
        self.state = self.gpcr.state
        self.embeddings = GetEmbeddings()

    def __len__(self) -> int:  # noqa: D105
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Tuple, Tuple]:  # noqa: D105
        row = self.data.iloc[[index],]  # type: ignore
        y_label = row[self.gpcr.y_col_name].values[0]
        y_label = torch.from_numpy(np.array(y_label, dtype=np.float32)).to(torch.float32)
        msa = row[self.gpcr.msa_col_name].values[0]
        smile = row[self.gpcr.selfies_col_name].values[0]
        canonical_smile = row[self.gpcr.canonical_smiles_col_name].values[0]
        morgan_fingerprint = row[self.gpcr.morgan_fp_col_name].values[0]
        label = row.label.values[0]
        uniprot_id = row.uniprot_id.values[0]

        try:
            smile_desc = literal_eval(row[self.gpcr.desc_col_name].values[0])
        except ValueError:
            smile_desc = row[self.gpcr.desc_col_name].values[0]  # if numpy array from parquet
        try:
            charge_fp = literal_eval(row[self.gpcr.charge_fp_col_name].values[0])
        except ValueError:
            charge_fp = row[self.gpcr.charge_fp_col_name].values[0]
        smile_desc = torch.tensor(smile_desc).float().reshape(1, -1)
        msa_int = self.ProteinTokenizer(
            msa, max_length=1900, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids[0]  # type: ignore

        msa_int = msa_int.long()
        emb = self.embeddings.get_embeddings(uniprot_id)
        emb = torch.tensor(emb).unsqueeze(0).to(torch.float32)
        chem_int = torch.Tensor(tokenize_smiles(canonical_smile)).unsqueeze(0)
        chem_int = get_pad_array(chem_int, (1, 100), -1)
        chem_int = torch.tensor(chem_int).long()
        mfp = torch.tensor(morgan_fingerprint).unsqueeze(0).to(torch.float32)
        charge_fp: Tensor = torch.tensor(charge_fp).unsqueeze(0)
        charge_morgan = torch.cat((charge_fp, mfp), dim=1)
        # assert not torch.isnan(msa_int).any(), "msa_int has nan"
        # assert not torch.isnan(chem_int).any(), "padded_chem_int has nan"
        # assert not torch.isnan(mfp).any(), "mfp has nan"
        # assert not torch.isnan(smile_desc).any(), "smile_desc has nan"
        # assert not torch.isnan(y_label).any(), "y_label has nan"
        return (msa_int, chem_int, smile_desc, charge_morgan, label, emb), (y_label, label)



def get_padded_graph_feat(output, max_len=64, other_len=1024):
    desired_shape = (max_len, other_len)
    pad_rows = max(0, desired_shape[0] - output.size(0))
    pad_cols = max(0, desired_shape[1] - output.size(1))
    padded_tensor = torch.nn.functional.pad(output, (0, pad_cols, 0, pad_rows))
    if padded_tensor.shape[0] > desired_shape[0]:
        padded_tensor = padded_tensor[: desired_shape[0], :]
    if padded_tensor.shape[1] > desired_shape[1]:
        padded_tensor = padded_tensor[:, : desired_shape[1]]
    return padded_tensor
