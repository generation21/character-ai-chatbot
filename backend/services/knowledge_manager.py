"""
지식 관리 모듈

FAISS를 사용하여 캐릭터 지식을 저장하고 검색합니다.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)

# 인덱스 파일명
FAISS_INDEX_NAME = "frieren_knowledge"


class KnowledgeManager:
    """FAISS 기반 캐릭터 지식 관리"""

    def __init__(self):
        self._vectorstore: Optional[FAISS] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._initialized = False

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """임베딩 모델 로드 (지연 로딩)"""
        if self._embeddings is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cuda"},  # GPU 사용
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Embedding model loaded successfully")
        return self._embeddings

    def _get_index_path(self) -> Path:
        """FAISS 인덱스 저장 경로"""
        return Path(settings.FAISS_DB_PATH) / FAISS_INDEX_NAME

    def _get_vectorstore(self) -> Optional[FAISS]:
        """FAISS 벡터스토어 로드 (지연 로딩)"""
        if self._vectorstore is None:
            index_path = self._get_index_path()

            if index_path.exists():
                logger.info(f"Loading existing FAISS index from {index_path}")
                try:
                    self._vectorstore = FAISS.load_local(
                        str(index_path),
                        self._get_embeddings(),
                        allow_dangerous_deserialization=True
                    )
                    self._initialized = True
                except Exception as e:
                    logger.warning(f"Failed to load FAISS index: {e}")
                    self._vectorstore = None
            else:
                logger.warning(f"FAISS index not found at {index_path}. Run index_knowledge.py first.")

        return self._vectorstore

    def index_pdf(self, pdf_path: Optional[str] = None, force: bool = False) -> int:
        """
        PDF 파일 또는 디렉토리의 모든 PDF를 청크로 분할하여 FAISS에 인덱싱

        Args:
            pdf_path: PDF 파일 또는 디렉토리 경로 (기본값: settings.PDF_DIR)
            force: 기존 데이터 삭제 후 재인덱싱

        Returns:
            인덱싱된 청크 수
        """
        pdf_path = Path(pdf_path or settings.PDF_DIR)

        if not pdf_path.exists():
            raise FileNotFoundError(f"Path not found: {pdf_path}")

        # PDF 파일 목록 수집
        if pdf_path.is_dir():
            pdf_files = list(pdf_path.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in: {pdf_path}")
            logger.info(f"Found {len(pdf_files)} PDF files in {pdf_path}")
        else:
            pdf_files = [pdf_path]

        # 모든 PDF에서 페이지 로드
        all_pages = []
        for pdf_file in pdf_files:
            logger.info(f"Loading PDF: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            # 문서에 소스 파일명 메타데이터 추가
            for page in pages:
                page.metadata["source_file"] = pdf_file.name
            all_pages.extend(pages)
            logger.info(f"  - Loaded {len(pages)} pages from {pdf_file.name}")

        logger.info(f"Total pages loaded: {len(all_pages)}")

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        chunks = text_splitter.split_documents(all_pages)
        logger.info(f"Split into {len(chunks)} chunks")

        # 저장 경로 준비
        index_path = self._get_index_path()

        # 강제 재인덱싱 시 기존 데이터 삭제
        if force and index_path.exists():
            import shutil
            shutil.rmtree(index_path)
            logger.info("Deleted existing FAISS index")
            self._vectorstore = None

        # FAISS 인덱스 생성
        index_path.parent.mkdir(parents=True, exist_ok=True)

        self._vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self._get_embeddings()
        )

        # 5. 인덱스 저장
        self._vectorstore.save_local(str(index_path))
        self._initialized = True

        logger.info(f"Indexed {len(chunks)} chunks to FAISS at {index_path}")
        return len(chunks)

    async def search(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        쿼리와 관련된 문서 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수 (기본값: settings.RAG_TOP_K)

        Returns:
            관련 문서 내용 리스트
        """
        top_k = top_k or settings.RAG_TOP_K
        vectorstore = self._get_vectorstore()

        if vectorstore is None:
            logger.warning("FAISS index not initialized. Run indexing first.")
            return []

        # 유사도 검색
        try:
            results = vectorstore.similarity_search(query, k=top_k)
            return [doc.page_content for doc in results]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get_context_for_prompt(self, query: str) -> str:
        """
        프롬프트에 삽입할 컨텍스트 문자열 생성

        Args:
            query: 사용자 메시지

        Returns:
            포맷팅된 컨텍스트 문자열
        """
        results = await self.search(query)

        if not results:
            return "관련 정보를 찾을 수 없습니다."

        # 검색 결과를 포맷팅
        context_parts = []
        for i, content in enumerate(results, 1):
            # 청크 내용 정리 (줄바꿈 제거, 공백 정리)
            cleaned = " ".join(content.split())
            context_parts.append(f"- {cleaned}")

        return "\n".join(context_parts)

    def get_collection_stats(self) -> dict:
        """컬렉션 상태 조회"""
        try:
            vectorstore = self._get_vectorstore()

            if vectorstore is None:
                return {
                    "error": "FAISS index not found",
                    "initialized": False,
                    "index_path": str(self._get_index_path())
                }

            # FAISS 인덱스 크기 확인
            doc_count = vectorstore.index.ntotal

            return {
                "index_name": FAISS_INDEX_NAME,
                "document_count": doc_count,
                "index_path": str(self._get_index_path()),
                "embedding_model": settings.EMBEDDING_MODEL,
                "initialized": self._initialized
            }
        except Exception as e:
            return {
                "error": str(e),
                "initialized": False
            }


# 싱글톤 인스턴스
knowledge_manager = KnowledgeManager()
