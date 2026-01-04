"""
세션 관리 모듈

SQLite 기반으로 대화 세션과 메시지를 관리합니다.
"""
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiosqlite

from config import settings
from schemas import MessageInfo, SessionInfo

logger = logging.getLogger(__name__)


class SessionManager:
    """SQLite 기반 세션 및 대화 기록 관리"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.DB_PATH
        self._initialized = False

    async def _ensure_db(self) -> None:
        """DB 파일과 테이블이 존재하는지 확인하고 없으면 생성"""
        if self._initialized:
            return

        # DB 디렉토리 생성
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # 세션 테이블
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_message_at TIMESTAMP,
                    summary TEXT
                )
            """)

            # 메시지 테이블
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    emotion_tag TEXT,
                    saturation_tag TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            # 인덱스 생성
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_created
                ON messages(session_id, created_at)
            """)

            await db.commit()

        self._initialized = True
        logger.info(f"Database initialized at {self.db_path}")

    async def create_session(self) -> str:
        """새 세션을 생성하고 session_id 반환"""
        await self._ensure_db()

        session_id = str(uuid.uuid4())
        now = datetime.now()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
                (session_id, now)
            )
            await db.commit()

        logger.info(f"Created new session: {session_id}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """세션 정보 조회"""
        await self._ensure_db()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # 세션 기본 정보
            cursor = await db.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,)
            )
            row = await cursor.fetchone()

            if not row:
                return None

            # 메시지 개수
            cursor = await db.execute(
                "SELECT COUNT(*) as count FROM messages WHERE session_id = ?",
                (session_id,)
            )
            count_row = await cursor.fetchone()
            message_count = count_row["count"] if count_row else 0

            return SessionInfo(
                session_id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
                last_message_at=datetime.fromisoformat(row["last_message_at"]) if row["last_message_at"] else None,
                message_count=message_count,
                summary=row["summary"]
            )

    async def session_exists(self, session_id: str) -> bool:
        """세션 존재 여부 확인"""
        await self._ensure_db()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT 1 FROM sessions WHERE id = ?",
                (session_id,)
            )
            return await cursor.fetchone() is not None

    async def list_sessions(self) -> List[SessionInfo]:
        """모든 세션 목록 조회 (최근순 정렬)"""
        await self._ensure_db()

        sessions = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT s.*, COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                GROUP BY s.id
                ORDER BY COALESCE(s.last_message_at, s.created_at) DESC
            """)

            rows = await cursor.fetchall()
            for row in rows:
                sessions.append(SessionInfo(
                    session_id=row["id"],
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
                    last_message_at=datetime.fromisoformat(row["last_message_at"]) if row["last_message_at"] else None,
                    message_count=row["message_count"],
                    summary=row["summary"]
                ))

        return sessions

    async def delete_session(self, session_id: str) -> bool:
        """세션 삭제 (관련 메시지도 함께 삭제)"""
        await self._ensure_db()

        async with aiosqlite.connect(self.db_path) as db:
            # 메시지 먼저 삭제
            await db.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,)
            )
            # 세션 삭제
            cursor = await db.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session_id,)
            )
            await db.commit()

            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted session: {session_id}")
            return deleted

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        emotion_tag: Optional[str] = None,
        saturation_tag: Optional[str] = None
    ) -> None:
        """메시지 추가"""
        await self._ensure_db()

        now = datetime.now()

        async with aiosqlite.connect(self.db_path) as db:
            # 메시지 추가
            await db.execute(
                """INSERT INTO messages
                   (session_id, role, content, emotion_tag, saturation_tag, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, role, content, emotion_tag, saturation_tag, now)
            )

            # 세션의 last_message_at 업데이트
            await db.execute(
                "UPDATE sessions SET last_message_at = ? WHERE id = ?",
                (now, session_id)
            )

            await db.commit()

    async def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[MessageInfo]:
        """세션의 메시지 조회 (오래된 순)"""
        await self._ensure_db()

        messages = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if limit:
                # 최근 N개 메시지 가져오기 (서브쿼리로 최근 것 가져온 후 정렬)
                cursor = await db.execute(
                    """SELECT * FROM (
                        SELECT * FROM messages
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                        LIMIT ? OFFSET ?
                    ) ORDER BY created_at ASC""",
                    (session_id, limit, offset)
                )
            else:
                cursor = await db.execute(
                    """SELECT * FROM messages
                       WHERE session_id = ?
                       ORDER BY created_at ASC""",
                    (session_id,)
                )

            rows = await cursor.fetchall()
            for row in rows:
                messages.append(MessageInfo(
                    role=row["role"],
                    content=row["content"],
                    emotion_tag=row["emotion_tag"],
                    saturation_tag=row["saturation_tag"],
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now()
                ))

        return messages

    async def get_message_count(self, session_id: str) -> int:
        """세션의 메시지 개수 조회"""
        await self._ensure_db()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,)
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def update_summary(self, session_id: str, summary: str) -> None:
        """세션 요약 업데이트"""
        await self._ensure_db()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE sessions SET summary = ? WHERE id = ?",
                (summary, session_id)
            )
            await db.commit()
            logger.info(f"Updated summary for session: {session_id}")

    async def get_summary(self, session_id: str) -> Optional[str]:
        """세션 요약 조회"""
        await self._ensure_db()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT summary FROM sessions WHERE id = ?",
                (session_id,)
            )
            row = await cursor.fetchone()
            return row[0] if row else None

    async def get_old_messages_for_summary(
        self,
        session_id: str,
        keep_recent: int
    ) -> List[MessageInfo]:
        """요약을 위해 오래된 메시지 조회 (최근 keep_recent개 제외)"""
        await self._ensure_db()

        messages = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # 전체 메시지 개수
            total_count = await self.get_message_count(session_id)

            if total_count <= keep_recent:
                return []

            # 오래된 메시지만 가져오기
            old_count = total_count - keep_recent
            cursor = await db.execute(
                """SELECT * FROM messages
                   WHERE session_id = ?
                   ORDER BY created_at ASC
                   LIMIT ?""",
                (session_id, old_count)
            )

            rows = await cursor.fetchall()
            for row in rows:
                messages.append(MessageInfo(
                    role=row["role"],
                    content=row["content"],
                    emotion_tag=row["emotion_tag"],
                    saturation_tag=row["saturation_tag"],
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now()
                ))

        return messages


# 싱글톤 인스턴스
session_manager = SessionManager()
