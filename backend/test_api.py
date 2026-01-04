"""
Conversation Memory System 테스트 스크립트

세션 관리 및 대화 기억 기능을 테스트합니다.
"""
import asyncio
import logging
import sys

import httpx

from config import settings
from schemas import ChatResponse
from services.comfyui_client import ComfyUIClient, comfyui_client
from services.image_prompt_generator import generate_image_prompt
from services.memory_manager import memory_manager
from services.session_manager import session_manager
from services.vllm_client import generate_response

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_session_manager() -> bool:
    """SessionManager 기능 테스트"""
    logger.info("🧪 SessionManager 테스트 시작...")

    try:
        # 1. 세션 생성
        session_id = await session_manager.create_session()
        logger.info(f"✅ 세션 생성: {session_id}")

        # 2. 세션 존재 확인
        exists = await session_manager.session_exists(session_id)
        assert exists, "세션이 존재해야 합니다"
        logger.info(f"✅ 세션 존재 확인: {exists}")

        # 3. 메시지 추가
        await session_manager.add_message(session_id, "human", "안녕하세요!")
        await session_manager.add_message(session_id, "ai", "안녕, 나는 프리렌이야.", "neutral", "0.3")
        logger.info("✅ 메시지 추가 완료")

        # 4. 메시지 조회
        messages = await session_manager.get_messages(session_id)
        assert len(messages) == 2, f"메시지 개수: {len(messages)}"
        logger.info(f"✅ 메시지 조회: {len(messages)}개")

        # 5. 세션 정보 조회
        session_info = await session_manager.get_session(session_id)
        assert session_info is not None
        assert session_info.message_count == 2
        logger.info(f"✅ 세션 정보: message_count={session_info.message_count}")

        # 6. 세션 목록 조회
        sessions = await session_manager.list_sessions()
        assert len(sessions) >= 1
        logger.info(f"✅ 세션 목록: {len(sessions)}개")

        # 7. 요약 업데이트
        await session_manager.update_summary(session_id, "테스트 요약입니다.")
        summary = await session_manager.get_summary(session_id)
        assert summary == "테스트 요약입니다."
        logger.info(f"✅ 요약 저장/조회 완료")

        # 8. 세션 삭제
        deleted = await session_manager.delete_session(session_id)
        assert deleted
        logger.info(f"✅ 세션 삭제 완료")

        # 삭제 확인
        exists_after = await session_manager.session_exists(session_id)
        assert not exists_after
        logger.info(f"✅ 삭제 확인: 세션이 더 이상 존재하지 않음")

        return True

    except Exception as e:
        logger.error(f"❌ SessionManager 테스트 실패: {e}")
        return False


async def test_memory_manager() -> bool:
    """MemoryManager 기능 테스트"""
    logger.info("\n🧪 MemoryManager 테스트 시작...")

    try:
        # 1. 세션 생성/조회
        session_id = await memory_manager.get_or_create_session(None)
        logger.info(f"✅ 세션 생성: {session_id}")

        # 2. 같은 세션 ID로 조회
        same_session = await memory_manager.get_or_create_session(session_id)
        assert same_session == session_id
        logger.info(f"✅ 기존 세션 재사용 확인")

        # 3. 유효하지 않은 세션 ID
        new_session = await memory_manager.get_or_create_session("invalid-session-id")
        assert new_session != "invalid-session-id"
        logger.info(f"✅ 유효하지 않은 세션 시 새 세션 생성: {new_session}")

        # 4. 대화 턴 저장
        await memory_manager.save_conversation_turn(session_id=session_id,
                                                    user_message="테스트 메시지",
                                                    ai_response="테스트 응답",
                                                    emotion_tag="neutral",
                                                    saturation_tag="0.5")
        logger.info("✅ 대화 턴 저장 완료")

        # 5. 프롬프트 메시지 구성 테스트
        prompt_messages = await memory_manager.build_prompt_messages(session_id, "새 메시지")
        logger.info(f"✅ 프롬프트 메시지 구성: {len(prompt_messages)}개")

        # 정리
        await session_manager.delete_session(session_id)
        await session_manager.delete_session(new_session)

        return True

    except Exception as e:
        logger.error(f"❌ MemoryManager 테스트 실패: {e}")
        return False


async def test_vllm_server_connection() -> bool:
    """vLLM 서버 연결 테스트"""
    logger.info(f"\n🔗 vLLM 서버 연결 테스트... ({settings.VLLM_API_URL})")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.VLLM_API_URL}/models")

            if response.status_code == 200:
                data = response.json()
                models = [m.get("id") for m in data.get("data", [])]
                logger.info(f"✅ vLLM 서버 연결 성공!")
                logger.info(f"📦 사용 가능한 모델: {models}")
                return True
            else:
                logger.error(f"❌ vLLM 서버 응답 오류: {response.status_code}")
                return False

    except httpx.ConnectError as e:
        logger.warning(f"⚠️ vLLM 서버에 연결할 수 없습니다: {e}")
        logger.info(f"💡 vLLM 서버가 실행 중이 아니면 generate_response 테스트를 건너뜁니다.")
        return False
    except Exception as e:
        logger.error(f"❌ 연결 테스트 중 오류: {e}")
        return False


async def test_generate_response_with_session() -> bool:
    """세션 기반 generate_response 테스트"""
    logger.info("\n🧪 generate_response (with session) 테스트...")

    try:
        # 첫 번째 대화 - 세션 자동 생성
        response1 = await generate_response("안녕, 프리렌!")
        logger.info(f"✅ 첫 번째 응답 (새 세션): {response1.session_id[:8]}...")
        logger.info(f"   💬 {response1.response[:50]}...")

        session_id = response1.session_id

        # 두 번째 대화 - 같은 세션 사용
        response2 = await generate_response("마법에 대해 알려줘.", session_id)
        assert response2.session_id == session_id
        logger.info(f"✅ 두 번째 응답 (같은 세션): {response2.session_id[:8]}...")
        logger.info(f"   💬 {response2.response[:50]}...")

        # 세션 정보 확인
        session_info = await session_manager.get_session(session_id)
        logger.info(f"✅ 세션 메시지 수: {session_info.message_count}")

        # 정리
        await session_manager.delete_session(session_id)

        return True

    except Exception as e:
        logger.error(f"❌ generate_response 테스트 실패: {e}")
        return False


async def test_comfyui_client() -> bool:
    """
    ComfyUI 클라이언트 테스트
    - 워크플로우 로드
    - 프롬프트 준비 (변수 치환)
    - 서버 연결 확인
    """
    logger.info("\n🧪 ComfyUI 클라이언트 테스트 시작...")

    try:
        # 1. 클라이언트 초기화 및 워크플로우 로드
        client = ComfyUIClient()
        assert client.workflow is not None, "워크플로우가 로드되어야 합니다"
        logger.info("✅ 워크플로우 로드 성공")

        # 2. 프롬프트 준비 테스트
        test_tags = "sitting, reading book, calm expression"
        prompt, seed, filename = client._prepare_prompt(test_tags)

        # 변수 치환 확인
        assert isinstance(seed, int), "seed는 정수여야 합니다"
        assert filename.startswith("frieren_"), "filename은 'frieren_'로 시작해야 합니다"
        logger.info(f"✅ 프롬프트 준비: filename={filename}, seed={seed}")

        # 3. JSON 내 변수 치환 확인
        import json
        prompt_str = json.dumps(prompt)
        assert "$positive_prompt" not in prompt_str, "$positive_prompt가 치환되어야 합니다"
        assert "$seed" not in prompt_str, "$seed가 치환되어야 합니다"
        assert "$file_name" not in prompt_str, "$file_name이 치환되어야 합니다"
        logger.info("✅ 변수 치환 확인 완료")

        # 4. base prompt 포함 확인
        assert "anime screencap" in prompt_str, "Base prompt가 포함되어야 합니다"
        assert "frieren" in prompt_str, "'frieren' 키워드가 포함되어야 합니다"
        assert test_tags in prompt_str, "추가 태그가 포함되어야 합니다"
        logger.info("✅ 프롬프트 내용 확인 완료")

        # 5. 서버 연결 확인 (선택적)
        is_connected = await client.check_connection()
        if is_connected:
            logger.info("✅ ComfyUI 서버 연결 확인")
        else:
            logger.info("⚠️ ComfyUI 서버 미연결 (선택적)")

        return True

    except Exception as e:
        logger.error(f"❌ ComfyUI 클라이언트 테스트 실패: {e}")
        return False


async def test_image_prompt_generator() -> bool:
    """
    이미지 프롬프트 생성기 테스트 (vLLM 필요)
    """
    logger.info("\n🧪 이미지 프롬프트 생성기 테스트 시작...")

    try:
        # 테스트 케이스들
        test_cases = [{
            "context": "그렇네... 바람이 시원해서 좋은 날씨일지도 모르지.",
            "emotion": "neutral",
            "saturation": "0.3"
        }, {
            "context": "하이터와 함께한 여행은... 좋은 추억이야.",
            "emotion": "happy",
            "saturation": "0.6"
        }, {
            "context": "히멜이... 떠나버렸네.",
            "emotion": "sad",
            "saturation": "0.8"
        }]

        for i, case in enumerate(test_cases, 1):
            prompt = await generate_image_prompt(conversation_context=case["context"],
                                                 emotion_tag=case["emotion"],
                                                 saturation_tag=case["saturation"])

            assert isinstance(prompt, str), "프롬프트는 문자열이어야 합니다"
            assert len(prompt) > 0, "프롬프트가 비어있으면 안됩니다"
            logger.info(f"✅ 테스트 케이스 {i}: {case['emotion']}")
            logger.info(f"   프롬프트: {prompt[:80]}..." if len(prompt) > 80 else f"   프롬프트: {prompt}")

        return True

    except Exception as e:
        logger.error(f"❌ 이미지 프롬프트 생성기 테스트 실패: {e}")
        return False


async def test_comfyui_connection() -> bool:
    """ComfyUI 서버 연결 테스트"""
    logger.info(f"\n🔗 ComfyUI 서버 연결 테스트... ({settings.COMFYUI_API_URL})")

    try:
        is_connected = await comfyui_client.check_connection()

        if is_connected:
            logger.info("✅ ComfyUI 서버 연결 성공!")
            return True
        else:
            logger.warning("⚠️ ComfyUI 서버에 연결할 수 없습니다")
            logger.info("💡 ComfyUI 서버가 실행 중이 아니면 이미지 생성 테스트를 건너뜁니다.")
            return False

    except Exception as e:
        logger.error(f"❌ 연결 테스트 중 오류: {e}")
        return False


async def run_all_tests() -> None:
    """모든 테스트 실행"""
    logger.info("=" * 60)
    logger.info("🚀 Conversation Memory System 테스트 시작")
    logger.info(f"📍 Database: {settings.DB_PATH}")
    logger.info(f"🔢 MAX_RECENT_MESSAGES: {settings.MAX_RECENT_MESSAGES}")
    logger.info(f"🔢 SUMMARIZE_THRESHOLD: {settings.SUMMARIZE_THRESHOLD}")
    logger.info("=" * 60)

    results = {}

    # 1. SessionManager 테스트
    results["SessionManager"] = await test_session_manager()

    # 2. MemoryManager 테스트
    results["MemoryManager"] = await test_memory_manager()

    # 3. vLLM 서버 연결 테스트
    vllm_connected = await test_vllm_server_connection()
    results["vLLM 서버 연결"] = vllm_connected

    # 4. generate_response 테스트 (vLLM 서버가 연결된 경우만)
    if vllm_connected:
        results["generate_response"] = await test_generate_response_with_session()
    else:
        logger.info("\n⏭️ vLLM 서버 미연결로 generate_response 테스트 건너뜀")
        results["generate_response"] = None

    # 5. 이미지 생성 관련 테스트
    results["ComfyUI 클라이언트"] = await test_comfyui_client()

    # 6. ComfyUI 서버 연결 테스트
    comfyui_connected = await test_comfyui_connection()
    results["ComfyUI 서버 연결"] = comfyui_connected

    # 7. 이미지 프롬프트 생성기 테스트 (vLLM 필요)
    if vllm_connected:
        results["이미지 프롬프트 생성기"] = await test_image_prompt_generator()
    else:
        logger.info("\n⏭️ vLLM 서버 미연결로 이미지 프롬프트 생성기 테스트 건너뜀")
        results["이미지 프롬프트 생성기"] = None

    # 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("📊 테스트 결과 요약")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        if passed is None:
            status = "⏭️ 건너뜀"
        elif passed:
            status = "✅ 통과"
        else:
            status = "❌ 실패"
            all_passed = False
        logger.info(f"  {test_name}: {status}")

    logger.info("=" * 60)

    if all_passed:
        logger.info("🎉 모든 테스트 통과!")
        sys.exit(0)
    else:
        logger.error("💥 일부 테스트 실패!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
