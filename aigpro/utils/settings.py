from os import getenv
from typing import Optional
# from pydantic import BaseSettings
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "aigpro_api"
    mode: Optional[str] = getenv("MODE", "dev")
    dbpath: Optional[str] = getenv("DBPATH", "aigpro_api/db/aigpro.db")

    class Config:
        env_file = f"aigpro_api/envs/{getenv('MODE')}.env"