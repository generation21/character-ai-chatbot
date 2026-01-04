from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    VLLM_API_URL: str = "http://localhost:8000/v1"
    MODEL_NAME: str = "./models/llm/Qwen3-4B-Instruct-2507"

    # Database settings
    DB_PATH: str = "./data/chat_history.db"

    # Memory settings
    MAX_RECENT_MESSAGES: int = 10  # 프롬프트에 포함할 최근 메시지 개수
    SUMMARIZE_THRESHOLD: int = 20  # 요약 트리거 임계값

    # RAG settings
    FAISS_DB_PATH: str = "./data/faiss_index"
    PDF_DIR: str = "./data/frieren"  # PDF 파일들이 있는 디렉토리
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RAG_TOP_K: int = 3  # 검색할 관련 문서 개수
    RAG_CHUNK_SIZE: int = 500  # 청크 크기 (문자)
    RAG_CHUNK_OVERLAP: int = 50  # 청크 오버랩

    # ComfyUI settings
    COMFYUI_API_URL: str = "http://localhost:8188"
    COMFYUI_WORKFLOW_PATH: str = "./data/comfyui_workflow/sdxl_api.json"
    COMFYUI_OUTPUT_DIR: str = "./data/generated_images"

    class Config:
        env_file = ".env"


settings = Settings()
