"""
ComfyUI API 클라이언트 모듈

ComfyUI 서버와 통신하여 이미지를 생성합니다.
"""
import json
import logging
import random
import uuid
from pathlib import Path
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

# 기본 프롬프트 (항상 포함)
BASE_POSITIVE_PROMPT = "anime screencap, masterpiece, best quality, frieren, 1girl, solo, ear blush, grey hair, long hair"


class ComfyUIClient:
    """ComfyUI API 클라이언트"""

    def __init__(self):
        self.api_url = settings.COMFYUI_API_URL
        self.workflow_path = Path(settings.COMFYUI_WORKFLOW_PATH)
        self.output_dir = Path(settings.COMFYUI_OUTPUT_DIR)
        self.workflow: Optional[dict] = None
        self._load_workflow()

    def _load_workflow(self) -> None:
        """워크플로우 JSON 로드"""
        try:
            with open(self.workflow_path, "r", encoding="utf-8") as f:
                self.workflow = json.load(f)
            logger.info(f"Workflow loaded from {self.workflow_path}")
        except FileNotFoundError:
            logger.error(f"Workflow file not found: {self.workflow_path}")
            self.workflow = None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse workflow JSON: {e}")
            self.workflow = None

    def _prepare_prompt(self, additional_tags: str, seed: Optional[int] = None, filename: Optional[str] = None) -> dict:
        """
        워크플로우 템플릿에 변수를 치환하여 프롬프트 준비

        Args:
            additional_tags: 추가할 프롬프트 태그들
            seed: 시드값 (None이면 랜덤 생성)
            filename: 파일명 prefix (None이면 UUID 생성)

        Returns:
            준비된 워크플로우 딕셔너리
        """
        if self.workflow is None:
            raise RuntimeError("Workflow not loaded")

        # 워크플로우 복사
        prompt = json.loads(json.dumps(self.workflow))

        # 변수 치환
        actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        actual_filename = filename if filename is not None else f"frieren_{uuid.uuid4().hex[:8]}"

        # 전체 positive prompt 구성
        full_prompt = f"{BASE_POSITIVE_PROMPT}, {additional_tags}" if additional_tags else BASE_POSITIVE_PROMPT

        # 노드별 변수 치환 (JSON 문자열로 변환 후 치환)
        prompt_str = json.dumps(prompt)
        prompt_str = prompt_str.replace("$positive_prompt", full_prompt)
        prompt_str = prompt_str.replace("\"$seed\"", str(actual_seed))
        prompt_str = prompt_str.replace("$file_name", actual_filename)

        return json.loads(prompt_str), actual_seed, actual_filename

    async def queue_prompt(self,
                           additional_tags: str,
                           seed: Optional[int] = None,
                           filename: Optional[str] = None,
                           timeout: float = 120.0) -> tuple[str, int]:
        """
        ComfyUI에 이미지 생성 요청

        Args:
            additional_tags: 추가 프롬프트 태그
            seed: 시드값
            filename: 파일명 prefix
            timeout: 요청 타임아웃 (초)

        Returns:
            (생성된 파일명, 사용된 시드)
        """
        prompt, actual_seed, actual_filename = self._prepare_prompt(additional_tags, seed, filename)

        async with httpx.AsyncClient(timeout=timeout) as client:
            # 프롬프트 큐잉
            response = await client.post(f"{self.api_url}/prompt", json={"prompt": prompt})
            response.raise_for_status()
            result = response.json()
            prompt_id = result.get("prompt_id")

            if not prompt_id:
                raise RuntimeError("Failed to get prompt_id from ComfyUI")

            logger.info(f"Queued prompt: {prompt_id}")

            # 완료 대기 (폴링)
            await self._wait_for_completion(client, prompt_id, timeout)

            return actual_filename, actual_seed

    async def _wait_for_completion(self, client: httpx.AsyncClient, prompt_id: str, timeout: float) -> None:
        """
        이미지 생성 완료 대기

        Args:
            client: httpx 클라이언트
            prompt_id: 프롬프트 ID
            timeout: 타임아웃 (초)
        """
        import asyncio

        poll_interval = 1.0
        elapsed = 0.0

        while elapsed < timeout:
            response = await client.get(f"{self.api_url}/history/{prompt_id}")
            response.raise_for_status()
            history = response.json()

            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "success":
                    logger.info(f"Image generation completed: {prompt_id}")
                    return
                elif status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI generation failed: {status}")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Image generation timed out after {timeout}s")

    async def check_connection(self) -> bool:
        """ComfyUI 서버 연결 확인"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.api_url}/system_stats")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"ComfyUI connection check failed: {e}")
            return False


# 싱글톤 인스턴스
comfyui_client = ComfyUIClient()
