from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import swifter

# from artifacts.v76.gpcrscan.models.mamba_models import BestGPCR  # as model
import torch
from rdkit import Chem
from sympy import true  # noqa: F401
from aigpro.chem.descriptors import compute_ligand_descriptors
from aigpro.chem.descriptors import generate_charge_fingerprint
from aigpro.chem.descriptors import morgan_fingerprint
from aigpro.chem.old_tokenizer import tokenize_smiles
from aigpro.chem.tokenizers import load_custom_tokenizer
from aigpro.utils.utilities import get_pad_array
from aigpro.utils.utilities import load_dataframe


def canonical_smile(smi):
    """Canonicalize the SMILES string.

    Args:
        smi (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception as e:
        print(e)
        return smi


def charge_fp(smi):
    """Generate charge fingerprint.

    Args:
        smi (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        return list(generate_charge_fingerprint(smi))
    except Exception as e:
        print(e)
        return None


def compute_desc(smi):
    """Compute the ligand descriptors.

    Args:
        smi (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        return compute_ligand_descriptors(smi)
    except Exception as e:
        print(e)
        return None


class DataPrep:
    """Prepare the data for prediction."""

    def __init__(
        self,
        smi: List[str] | str = None,
        gpcr_protein_tokenizer_filename=None,
        msa_filename=None,
        canonical_smile_col="canonical_smiles",
        uniprot_id_col="uniprot_id",
        global_scan: bool = False,
    ) -> None:
        if isinstance(smi, list):
            self.smi_df = pd.DataFrame({"smiles": smi})
        else:
            self.smi_df = load_dataframe(smi)
        self.global_scan = global_scan
        # print(canonical_smile(smi[0]))

        if gpcr_protein_tokenizer_filename is not None:
            self.gpcr_protein_tokenizer_filename = gpcr_protein_tokenizer_filename

        else:
            gpcr_protein_tokenizer_filename = "GPCR_prot_tokenizer.json"
            # self.gpcr_protein_tokenizer_filename = Path(__file__).parent / gpcr_protein_tokenizer_filename
            # self.gpcr_protein_tokenizer_filename = Path("./references") / gpcr_protein_tokenizer_filename
            try:
                self.gpcr_protein_tokenizer_filename = (
                    Path(__file__).parent / "references" / gpcr_protein_tokenizer_filename
                )
                assert self.gpcr_protein_tokenizer_filename.exists(), f"{self.gpcr_protein_tokenizer_filename} does not exist. Next try to load from the current directory."

            except Exception as e:
                self.gpcr_protein_tokenizer_filename = Path("./references") / gpcr_protein_tokenizer_filename

            # self.gpcr_protein_tokenizer_filename = (
            #     Path(__file__).parent / "references" / gpcr_protein_tokenizer_filename
            # )

        if msa_filename is None:
            msa_filename = "MSA_DF.parquet"
            # msa_filename = Path(__file__).parent / msa_filename
            # msa_filename = Path("./references") / msa_filename
            msa_filename = Path(__file__).parent / "references" / msa_filename

            self.msa_df = load_dataframe(msa_filename)
        else:
            self.msa_df = load_dataframe(msa_filename)

        self.canonical_smile_col = canonical_smile_col
        self.uniprot_id_col = uniprot_id_col
        # print(self.msa_df.head())

        self.ProteinTokenizer = load_custom_tokenizer(
            self.gpcr_protein_tokenizer_filename,
        )
        self.__add_data()

    def __add_data(self):
        self.add_msa_token()
        self.get_canonical_smiles()
        self.add_mfp()
        self.add_chemical_token()
        self.get_charge_fp()
        self.get_desc()
        self.add_msa()

    def get_msa_token(self, uniprot_id):
        """Get the MSA token.

        Args:
            uniprot_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        msa = self.msa_df[self.msa_df[self.uniprot_id_col] == uniprot_id].protein_seq.values[0]
        msa_int = self.ProteinTokenizer(
            msa,
            max_length=1900,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        ).input_ids[0]
        return msa_int

    def add_msa_token(self):
        """Add the MSA token to the dataframe."""
        self.msa_df["msa_int"] = self.msa_df.swifter.allow_dask_on_strings(enable=True).apply(
            lambda x: self.get_msa_token(x[self.uniprot_id_col]), axis=1
        )

    def get_chem_token(self, canonical_smile):
        """Get the chemical token.

        Args:
            canonical_smile (_type_): _description_

        Returns:
            _type_: _description_
        """
        chem_int = torch.Tensor(tokenize_smiles(canonical_smile)).unsqueeze(0)
        chem_int = get_pad_array(chem_int, (1, 100), -1)
        # chem_int = torch.tensor(chem_int).long()
        return chem_int

    def add_chemical_token(self):
        """Add the chemical token to the dataframe."""
        self.smi_df["chem_int"] = self.smi_df.swifter.allow_dask_on_strings(enable=True).apply(
            lambda x: tokenize_smiles(x[self.canonical_smile_col]), axis=1
        )

    @property
    def df(self):
        return self.smi_df

    def get_canonical_smiles(self):
        """Get the canonical SMILES."""
        if self.canonical_smile_col in self.smi_df.columns:
            self.smi_df = self.smi_df.drop(columns=[self.canonical_smile_col])

        self.smi_df[self.canonical_smile_col] = self.smi_df.swifter.allow_dask_on_strings(enable=True).apply(
            lambda x: canonical_smile(x["smiles"]), axis=1
        )

    def get_charge_fp(self):
        """Get the charge fingerprint."""
        self.smi_df["charge_fp"] = self.smi_df.swifter.allow_dask_on_strings(enable=True).apply(
            lambda x: charge_fp(x[self.canonical_smile_col]), axis=1
        )

    def get_desc(self):
        """Get the ligand descriptors."""
        self.smi_df["desc"] = self.smi_df.swifter.allow_dask_on_strings(enable=True).apply(
            lambda x: compute_desc(x[self.canonical_smile_col]), axis=1
        )

    def add_msa(self):
        """Add the MSA to the dataframe."""
        _all_df = pd.DataFrame()
        _global_df = deepcopy(self.smi_df)
        if self.global_scan:
            for smi in _global_df[self.canonical_smile_col]:
                _tmp_df = _global_df[_global_df[self.canonical_smile_col] == smi]
                _tmp_df = _tmp_df.loc[_tmp_df.index.repeat(len(self.msa_df))].reset_index(drop=True)
                _tmp_df = pd.concat([_tmp_df, self.msa_df.copy()], axis=1, join="outer")
                _all_df = pd.concat([_all_df, _tmp_df], axis=0)
        else:
            _all_df = deepcopy(_global_df)
            assert "uniprot_id" in _all_df.columns, "uniprot_id not in the input dataframe"
            ## add msa row for matching uniprot_id
            _all_df = pd.merge(
                _all_df,
                self.msa_df,
                left_on="uniprot_id",
                right_on="uniprot_id",
                how="left",
            )

            # _all_df = pd.concat([_all_df, self.msa_df.copy()], axis=1, join="outer")
        if "label" in _all_df.columns:
            _all_df = _all_df.rename(columns={"label": "original_label"})

        _all_df["label"] = 1
        _all_df_2 = deepcopy(_all_df)
        _all_df_2["label"] = 0
        _all_df = pd.concat([_all_df, _all_df_2], axis=0)

        self.smi_df = _all_df.reset_index(drop=True)
        # self.smi_df = _all_df.reset_index(drop=True)
        # self.smi_df["Unit"] = ["nM"]
        ## add Standard Value

    def add_mfp(self):
        """Add the Morgan fingerprint to the dataframe."""
        self.smi_df["mfp"] = self.smi_df.swifter.allow_dask_on_strings(enable=True).apply(
            lambda x: morgan_fingerprint(x[self.canonical_smile_col]), axis=1
        )

    @staticmethod
    def canonical_smile(smi):
        """Canonicalize the SMILES string.

        Args:
            smi (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception as e:
            print(e)
            return smi

    @staticmethod
    def charge_fp(smi):
        """Generate charge fingerprint."""
        try:
            return list(generate_charge_fingerprint(smi))
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def compute_desc(smi):
        """Compute the ligand descriptors.

        Args:
            smi (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            return compute_ligand_descriptors(smi)
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def compute_mfp(smi):
        """Compute the Morgan fingerprint.

        Args:
            smi (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            return list(morgan_fingerprint(smi))
        except Exception as e:
            print(e)
            return None


class PredictDataloader(torch.utils.data.Dataset):
    """Data loader for prediction.

    Args:
        torch (_type_): _description_
    """

    def __init__(self, data: pd.DataFrame, device="cuda"):
        self.data = data
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        columns = ["msa_int", "chem_int", "desc", "charge_fp", "label", "mfp"]
        assert all([x in self.data.columns for x in columns]), f"columns not found: {columns}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        protein = torch.tensor(row.msa_int).long().to(self.device)
        ligand = torch.tensor(row.chem_int).long().to(self.device)
        smi_desc = torch.tensor(row.desc).float().to(self.device)
        charge_fp = torch.tensor(row.charge_fp).float().to(self.device).unsqueeze(0)
        clabel = torch.tensor(row.label).long().to(self.device)
        mfp = torch.tensor(row.mfp).float().to(self.device).unsqueeze(0)
        charge_morgan = torch.cat([charge_fp, mfp], dim=-1)
        return protein, ligand, smi_desc, charge_morgan, clabel, clabel


@dataclass
class PredResult:
    agonist: pd.DataFrame
    antagonist: pd.DataFrame
    combined: pd.DataFrame
    raw: pd.DataFrame

    def __repr__(self):
        return f"Agonist: {self.agonist.shape}, Antagonist: {self.antagonist.shape}, Combined: {self.combined.shape}, Raw: {self.raw.shape}"


class AiGProScript:
    def __init__(self, script_file, device="cuda", batch_size=128):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = torch.jit.load(script_file)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.batch_size = batch_size

    @torch.no_grad()
    def predict(self, test_data):
        loaded_data = self.data_loader(test_data)
        # test_data = [x.to(self.device) for x in test_data]

        predictions = []
        for i in loaded_data:
            pred = self.model(i)
            predictions.append(pred.cpu().numpy().flatten())

        return np.concatenate(predictions).flatten()
        # y_hat = self.model(loaded_data)
        # return y_hat.flatten().cpu().numpy()

    def data_loader(self, data_frame):
        return torch.utils.data.DataLoader(
            PredictDataloader(data_frame, device=None),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def df_predict(self, data_frame):
        predictions = self.predict(data_frame)
        data_frame["pred"] = predictions

        _raw = deepcopy(data_frame)

        agonist = data_frame[data_frame["label"] == 1].reset_index(drop=True)
        agonist["label"] = [1] * agonist.shape[0]
        agonist = agonist.rename(columns={"pred": "ago_pred"})
        # agonist = agonist.drop(columns=["label"])

        antagonist = data_frame[data_frame["label"] == 0].reset_index(drop=True)
        antagonist["label"] = [0] * antagonist.shape[0]
        antagonist = antagonist.rename(columns={"pred": "anta_pred"})
        # antagonist = antagonist.drop(columns=["label"])
        drop_cols = [
            "msa_int",
            "chem_int",
            "desc",
            "charge_fp",
            "mfp",
            "protein_seq",
            # "canonical_smiles",
            "bananas",
        ]
        # remove _x and _y
        for col in drop_cols:
            if col in agonist.columns:
                agonist = agonist.drop(columns=[col])
            if col in antagonist.columns:
                antagonist = antagonist.drop(columns=[col])
        combined = pd.merge(agonist, antagonist, on=["canonical_smiles", "uniprot_id"], how="outer")

        # remove all _y and rename _x to original
        for col in combined.columns:
            if col.endswith("_x"):
                combined = combined.rename(columns={col: col.replace("_x", "")})
            if col.endswith("_y"):
                combined = combined.drop(columns=[col])
        combined = combined.drop(columns=["label"])

        return PredResult(
            agonist,
            antagonist,
            combined.reset_index(drop=True),
            _raw,
        )


def prepare_data(smiles: list[str], global_scan=True) -> pd.DataFrame:
    """Prepare the data for prediction.

    Args:
        smiles (list[str]): _description_
        global_scan (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    if not isinstance(smiles, list):
        smiles = [smiles]
    data_prep = DataPrep(smi=smiles, global_scan=global_scan)
    return data_prep.df


def prediction_workflow(smiles: list[str], global_sca=True) -> pd.DataFrame:
    """Compute the prediction for the given SMILES.

    Args:
        smiles (list[str]): _description_
        global_sca (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    prepared_data = prepare_data(smiles, global_scan=global_sca)
    # model_path = Path(__file__).parent / "model_aigpro.pt"
    model_filename = "model_aigpro.pt"
    try:
        model_path = Path(__file__).parent / "references" / model_filename
        assert model_path.exists(), f"{model_path} does not exist. Next try to load from the current directory."
    except Exception as e:
        model_path = Path("./references") / model_filename

    predictor = AiGProScript(model_path)
    result = predictor.df_predict(prepared_data)
    return result.combined
