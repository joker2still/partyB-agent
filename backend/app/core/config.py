from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="qwen2.5", alias="OLLAMA_MODEL")
    api_base_url: str | None = Field(default=None, alias="API_BASE_URL")
    api_key: str | None = Field(default=None, alias="API_KEY")
    api_model: str | None = Field(default=None, alias="API_MODEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
