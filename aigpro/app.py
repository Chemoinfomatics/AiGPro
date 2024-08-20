import ipaddress
import logging
from pathlib import Path
from pyexpat import model
import re
import time
from typing import Annotated, Dict, List, Union
from typing import Optional
import pandas as pd
import swifter
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler
from aigpro.chem.descriptors import custom_add_endpoint
from aigpro.core.metrics import pic50_to_ic50
from aigpro.production_preprocess import prediction_workflow
from aigpro.utils.settings import Settings
import os
import json
from fastapi import exceptions
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from dotenv import load_dotenv

console = Console()

# log write to file
current_dir = Path(__file__).parent.absolute()
log_file = current_dir / "logs/aigpro.log"

Path(log_file).parent.mkdir(parents=True, exist_ok=True)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
# logger.addHandler(file_handler)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True), file_handler],
)

logger = logging.getLogger("rich")


from fastapi.security import OAuth2PasswordBearer
from fastapi import FastAPI, Depends, HTTPException, status


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # use token authentication
from dotenv import load_dotenv

load_dotenv()


def load_api_keys():
    current_dir = Path(__file__).parent.absolute()
    toekens_allowed_file = current_dir / "allowed_api_tokens.json"
    with open(toekens_allowed_file, "r") as file:
        users = json.load(file)

    # Set environment variables
    api_keys = []
    for user in users:
        env_var_name = f"API_TOKEN_{user['first_name'].upper()}"
        os.environ[env_var_name] = user["api_token"]
        api_keys.append(user["api_token"])
    return api_keys


api_keys = load_api_keys()


def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key not in api_keys:
        logger.error(f"API Key: '{api_key}' not in the list of allowed API keys {api_keys}.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden. You are authorized to access this resource.Email to rahul@drugonix.com to get the API key.",
        )
    logger.info(f"API Key: '{api_key}' is valid.")


app = FastAPI(
    title="AiGPro API",
    description="This API is used for AiGPro predictions for ranking of SMILES agianst the Human GPCRs.",
    contact={
        "name": "Rahul Brahma",
        "url": "https://kinscan.drugonix.com",
        "email": "rahul@drugonix.com",
    },
    root_path="/api/v1/",
)


settings = Settings()
origins = [
    "http://localhost:3000",
    "*",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SmilesInput(BaseModel):  # noqa: D101
    smile: str | List[str]
    glob_scan: Optional[bool] = True
    aig_version: Optional[str] = "PRO"  # PRO | ANT | AGO


from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm


@app.exception_handler(exceptions.RequestValidationError)
@app.exception_handler(ValidationError)
@app.post(
    "/predictaigpro",
    tags=["AiGPro API for ranking of SMILES against the Human Kinome"],
    dependencies=[Depends(api_key_auth)],
    response_model=Union[str, Dict, None],
)
async def predict(request: Request, incomingdata: SmilesInput):
    #   token: Annotated[str, Depends(oauth2_scheme)] = None):
    """Predict the activity value for GPCRS.

    Args:
        smiles_input (SmilesInput): _description_

    Returns:
        _type_: _description_
    """

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        smiles_input = SmilesInput(**body)
    except Exception as e:
        logger.error(f"Error in parsing the request: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

    logger.info(f"Received {len(smiles_input.smile)} SMILES for prediction.")

    try:
        return handle_aigpro_prediction(
            smiles_input.smile,
            # uniprot_id=smiles_input.target,
            global_scan=smiles_input.glob_scan,
            # model_version=smiles_input.model_version,
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"message": "Internal Server Error"}


def handle_aigpro_prediction(smiles, uniprot_id="P00533", global_scan=True):
    """_summary_.

    Args:
        smiles (_type_): _description_
        uniprot_id (str, optional): _description_. Defaults to "P00533".
        global_scan (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    try:
        logger.info(f"Glob Scan: {global_scan}")

        results = prediction_workflow(smiles, global_scan)
        results["ago_prob"] = results.swifter.apply(compute_azos_metrics, axis=1, args=("ago_pred",))
        results["anta_prob"] = results.swifter.apply(compute_azos_metrics, axis=1, args=("anta_pred",))

        ## remove space in column names
        results.columns = results.columns.str.replace(" ", "_")
        return results.to_json(orient="records")

    except Exception as e:
        logger.error(f"Error in handle_aigpro_optimizations: {str(e)}")
        raise


def compute_azos_metrics(row, colname="plog_endpoint"):  # noqa: D103
    _standard_value = pic50_to_ic50(row[colname], unit="nM")
    row["standard_value"] = _standard_value
    _value = custom_add_endpoint(row)
    return _value


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8097, reload=True, workers=4)
