#!/usr/bin/env python
"""
PDF 인덱싱 스크립트

사용법:
    uv run python scripts/index_knowledge.py
    uv run python scripts/index_knowledge.py --force  # 강제 재인덱싱
"""
import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from services.knowledge_manager import knowledge_manager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="PDF 지식 베이스 인덱싱")
    parser.add_argument("--force", "-f", action="store_true", help="기존 데이터 삭제 후 강제 재인덱싱")
    parser.add_argument("--path", type=str, default=None, help=f"PDF 파일 또는 디렉토리 경로 (기본값: {settings.PDF_DIR})")

    args = parser.parse_args()

    pdf_path = args.path or settings.PDF_DIR

    logger.info("=" * 60)
    logger.info("🚀 PDF 지식 베이스 인덱싱 시작")
    logger.info(f"📄 PDF 경로: {pdf_path}")
    logger.info(f"💾 FAISS 경로: {settings.FAISS_DB_PATH}")
    logger.info(f"🤖 임베딩 모델: {settings.EMBEDDING_MODEL}")
    logger.info(f"📏 청크 크기: {settings.RAG_CHUNK_SIZE} (오버랩: {settings.RAG_CHUNK_OVERLAP})")
    logger.info(f"🔄 강제 재인덱싱: {args.force}")
    logger.info("=" * 60)

    # 경로 존재 확인
    if not Path(pdf_path).exists():
        logger.error(f"❌ 경로를 찾을 수 없습니다: {pdf_path}")
        sys.exit(1)

    try:
        # 인덱싱 실행
        chunk_count = knowledge_manager.index_pdf(pdf_path, force=args.force)

        logger.info("=" * 60)
        logger.info(f"✅ 인덱싱 완료!")
        logger.info(f"📊 인덱싱된 청크 수: {chunk_count}")
        logger.info("=" * 60)

        # 통계 출력
        stats = knowledge_manager.get_collection_stats()
        logger.info(f"📈 컬렉션 통계: {stats}")

    except Exception as e:
        logger.error(f"❌ 인덱싱 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
