import json
import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    # Storage / queues
    input_container: str = Field(..., env="INPUT_CONTAINER")
    output_container: str = Field(..., env="OUTPUT_CONTAINER")
    poison_container: str = Field(..., env="POISON_CONTAINER")
    queue_name: str = Field(..., env="QUEUE_NAME")

    # Models and taxonomy
    model_version: str = Field("v1", env="MODEL_VERSION")
    taxonomy_path: str = Field(default="taxonomy_v1.json", env="TAXONOMY_PATH")
    allowed_languages: List[str] = Field(default_factory=lambda: ["en", "es"])

    # Optional paths / keys
    spam_keywords_path: Optional[str] = Field(None, env="SPAM_KEYWORDS_PATH")
    key_vault_url: Optional[str] = Field(None, env="KEY_VAULT_URL")
    app_insights_key: Optional[str] = Field(None, env="APP_INSIGHTS_INSTRUMENTATIONKEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("allowed_languages", pre=True)
    def parse_langs(cls, value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return [lang.strip() for lang in value.split(",") if lang.strip()]
        return ["en", "es"]


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]

