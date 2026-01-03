from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    VLLM_API_URL: str = "http://localhost:8000/v1"
    MODEL_NAME: str = "Qwen/Qwen3-4B-Instruct"

    class Config:
        env_file = ".env"

settings = Settings()
