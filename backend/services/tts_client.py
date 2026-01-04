"""
GPT-SoVITS TTS API 클라이언트 모듈

GPT-SoVITS 서버와 통신하여 음성을 생성합니다.
"""
import logging
import uuid
from pathlib import Path
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)


class TTSClient:
    """GPT-SoVITS TTS API 클라이언트"""

    def __init__(self) -> None:
        self.api_url = settings.TTS_API_URL
        # Resolve paths to absolute (relative paths are from project root)
        self.model_dir = Path(settings.TTS_MODEL_DIR).resolve()
        self.ref_audio_dir = Path(settings.TTS_REF_AUDIO_DIR).resolve()
        self.output_dir = Path(settings.TTS_OUTPUT_DIR).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model paths (absolute)
        self.gpt_weights = self.model_dir / settings.TTS_GPT_WEIGHTS
        self.sovits_weights = self.model_dir / settings.TTS_SOVITS_WEIGHTS

        # Emotion to reference audio mapping
        self.emotion_ref_map = {
            "neutral": "default_jp",
            "happy": "happy_jp",
            "sad": "sad_jp",
            "angry": "default_jp",  # fallback to default
            "surprised": "default_jp",
            "embarrassed": "default_jp",
        }

        self._initialized = False

    def _get_ref_audio_info(self, emotion: str = "neutral") -> tuple[Path, str]:
        """
        감정에 맞는 참조 오디오와 프롬프트 텍스트 반환

        Args:
            emotion: 감정 태그

        Returns:
            (ref_audio_path, prompt_text)
        """
        ref_name = self.emotion_ref_map.get(emotion, "default_jp")
        ref_audio = self.ref_audio_dir / f"{ref_name}.wav"
        ref_text_file = self.ref_audio_dir / f"{ref_name}.txt"

        # 프롬프트 텍스트 로드
        prompt_text = ""
        if ref_text_file.exists():
            prompt_text = ref_text_file.read_text(encoding="utf-8").strip()

        return ref_audio, prompt_text

    async def initialize_model(self) -> bool:
        """
        GPT-SoVITS 모델 가중치 초기화

        서버가 이미 모델을 로드한 상태일 수 있으므로 실패해도 계속 진행.

        Returns:
            성공 여부
        """
        if self._initialized:
            return True

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # GPT 가중치 설정 시도
                try:
                    resp = await client.get(f"{self.api_url}/set_gpt_weights",
                                            params={"weights_path": str(self.gpt_weights)})
                    if resp.status_code != 200:
                        logger.warning(f"GPT weights setting returned non-200: {resp.text}")
                        # 실패해도 계속 진행 - 서버에 이미 로드된 모델 사용
                except Exception as e:
                    logger.warning(f"GPT weights setting failed: {e}")

                # SoVITS 가중치 설정 시도
                try:
                    resp = await client.get(f"{self.api_url}/set_sovits_weights",
                                            params={"weights_path": str(self.sovits_weights)})
                    if resp.status_code != 200:
                        logger.warning(f"SoVITS weights setting returned non-200: {resp.text}")
                        # 실패해도 계속 진행 - 서버에 이미 로드된 모델 사용
                except Exception as e:
                    logger.warning(f"SoVITS weights setting failed: {e}")

                # 서버가 연결되어 있으면 초기화 완료로 표시
                self._initialized = True
                logger.info("TTS client initialized (using server's current model)")
                return True

        except Exception as e:
            logger.error(f"Failed to connect to TTS server: {e}")
            return False

    async def generate_audio(self,
                             text: str,
                             emotion: str = "neutral",
                             filename: Optional[str] = None,
                             text_lang: str = "ja",
                             timeout: float = 60.0) -> tuple[str, Path]:
        """
        텍스트를 음성으로 변환

        Args:
            text: 합성할 텍스트
            emotion: 감정 태그 (참조 오디오 선택에 사용)
            filename: 저장할 파일명 (없으면 UUID 생성)
            text_lang: 텍스트 언어 (ko, ja, zh, en)
            timeout: 요청 타임아웃

        Returns:
            (파일명, 파일 경로)
        """
        # 모델 초기화 확인
        if not self._initialized:
            await self.initialize_model()

        # 참조 오디오 및 프롬프트 가져오기
        ref_audio, prompt_text = self._get_ref_audio_info(emotion)

        # 파일명 생성
        actual_filename = filename or f"frieren_audio_{uuid.uuid4().hex[:8]}"
        output_path = self.output_dir / f"{actual_filename}.wav"

        # TTS 요청 구성
        request_data = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": str(ref_audio),
            "prompt_text": prompt_text,
            "prompt_lang": "ja",
            "top_k": 15,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": "cut5",
            "batch_size": 1,
            "speed_factor": 1.0,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "sample_steps": 32,  # v4 모델용
            "super_sampling": False,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.api_url}/tts", json=request_data)
            response.raise_for_status()

            # WAV 파일 저장
            output_path.write_bytes(response.content)
            logger.info(f"Audio generated: {output_path}")

            return actual_filename, output_path

    async def check_connection(self) -> bool:
        """TTS 서버 연결 확인"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # /tts 엔드포인트가 없으므로 간단한 요청으로 확인
                response = await client.get(f"{self.api_url}/control")
                # 400이라도 서버가 응답하면 연결된 것
                return response.status_code in [200, 400]
        except Exception as e:
            logger.warning(f"TTS connection check failed: {e}")
            return False


# 싱글톤 인스턴스
tts_client = TTSClient()
