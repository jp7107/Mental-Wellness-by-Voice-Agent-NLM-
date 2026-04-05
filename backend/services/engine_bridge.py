# ============================================
# MIND EASE — Engine Bridge (Python ↔ C++ IPC)
# ============================================
# Manages the C++ engine subprocess.
# Communication via stdin/stdout with length-prefixed
# binary frames (4-byte LE header + JSON/binary payload).

import asyncio
import json
import struct
import logging
from typing import Optional, Callable, Awaitable

from config import settings, PROJECT_ROOT

logger = logging.getLogger("mindease.engine_bridge")


class EngineBridge:
    """Manages the C++ inference engine as a subprocess."""

    def __init__(self):
        self._process: Optional[asyncio.subprocess.Process] = None
        self._on_message: Optional[Callable[[dict], Awaitable[None]]] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()

    async def start(self) -> bool:
        """Start the C++ engine subprocess."""
        engine_path = settings.resolve_path(settings.engine_binary_path)

        if not engine_path.exists():
            logger.warning(
                f"Engine binary not found at {engine_path} — running in mock mode"
            )
            self._ready.set()
            return True

        logger.info(f"Starting engine: {engine_path}")

        try:
            self._process = await asyncio.create_subprocess_exec(
                str(engine_path),
                "--config",
                str(settings.resolve_path(settings.pipeline_config_path)),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,  # 1MB buffer
                cwd=str(PROJECT_ROOT),
            )

            # Start reading output
            self._reader_task = asyncio.create_task(self._read_loop())

            # Start reading stderr (logs)
            asyncio.create_task(self._stderr_reader())

            # Wait for ready signal
            try:
                await asyncio.wait_for(self._ready.wait(), timeout=30.0)
                logger.info("Engine is ready")
                return True
            except asyncio.TimeoutError:
                logger.error("Engine did not become ready within 30s")
                await self.stop()
                return False

        except Exception as e:
            logger.error(f"Failed to start engine: {e}")
            return False

    async def stop(self):
        """Stop the engine subprocess."""
        if self._process and self._process.returncode is None:
            try:
                await self._send_raw(json.dumps({"type": "shutdown"}))
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                self._process.kill()
                await self._process.wait()

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        self._process = None
        logger.info("Engine stopped")

    async def send_audio(self, pcm_data: bytes):
        """Send audio data to the engine."""
        if not self._process:
            # Mock mode: generate a mock response
            await self._handle_mock_audio(pcm_data)
            return

        # Send audio header
        header_msg = json.dumps({"type": "audio"})
        await self._send_raw(header_msg)
        # Send raw audio data
        await self._send_raw_bytes(pcm_data)

    def set_message_handler(self, handler: Callable[[dict], Awaitable[None]]):
        """Set the callback for messages from the engine."""
        self._on_message = handler

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    # ── Private methods ──

    async def _send_raw(self, data: str):
        """Send a length-prefixed string message."""
        if not self._process or not self._process.stdin:
            return
        encoded = data.encode("utf-8")
        header = struct.pack("<I", len(encoded))
        self._process.stdin.write(header + encoded)
        await self._process.stdin.drain()

    async def _send_raw_bytes(self, data: bytes):
        """Send length-prefixed binary data."""
        if not self._process or not self._process.stdin:
            return
        header = struct.pack("<I", len(data))
        self._process.stdin.write(header + data)
        await self._process.stdin.drain()

    async def _read_loop(self):
        """Continuously read messages from the engine's stdout."""
        try:
            while self._process and self._process.returncode is None:
                # Read 4-byte length header
                header = await self._process.stdout.readexactly(4)
                length = struct.unpack("<I", header)[0]

                if length == 0 or length > 10 * 1024 * 1024:
                    logger.error(f"Invalid message length: {length}")
                    continue

                # Read message body
                data = await self._process.stdout.readexactly(length)
                message = json.loads(data.decode("utf-8"))

                # Handle status messages internally
                if message.get("type") == "status" and message.get("status") == "ready":
                    self._ready.set()

                # Forward to handler
                if self._on_message:
                    await self._on_message(message)

        except asyncio.IncompleteReadError:
            logger.info("Engine stdout closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Engine read error: {e}")

    async def _stderr_reader(self):
        """Forward engine stderr to Python logger."""
        try:
            while self._process and self._process.returncode is None:
                if self._process.stderr:
                    line = await self._process.stderr.readline()
                    if not line:
                        break
                    logger.debug(f"[engine] {line.decode().rstrip()}")
        except Exception:
            pass

    async def _handle_mock_audio(self, pcm_data: bytes):
        """Generate mock responses when engine binary is not available."""
        import random

        emotions = ["calm", "sad", "anxious", "angry", "fearful"]
        templates = {
            "calm": "It sounds like you're doing well. I'm here if you'd like to talk about anything.",
            "sad": "I hear some sadness in your words. Would you like to tell me more about what's on your mind?",
            "anxious": "I sense some anxiety. Let's take a moment — try a slow, deep breath with me.",
            "angry": "It sounds like something has upset you. Your feelings are valid — would you like to talk about it?",
            "fearful": "I can tell you might be feeling scared. You're safe here, and I'm listening.",
        }

        emotion = random.choice(emotions)
        response_text = templates[emotion]

        if self._on_message:
            # Simulate pipeline output
            await self._on_message(
                {
                    "type": "transcript",
                    "text": f"[mock] Audio received ({len(pcm_data)} bytes)",
                    "final": True,
                    "duration_ms": 50,
                }
            )
            await self._on_message(
                {
                    "type": "emotion",
                    "label": emotion,
                    "confidence": round(random.uniform(0.5, 0.95), 2),
                    "scores": {
                        e: round(random.uniform(0.05, 0.3), 4) for e in emotions
                    },
                    "duration_ms": 10,
                }
            )
            await self._on_message(
                {
                    "type": "mood_update",
                    "score": random.randint(1, 4),
                    "window_average": 2.0,
                    "window": [2, 2, 2],
                    "safety_triggered": False,
                }
            )
            await self._on_message(
                {
                    "type": "response",
                    "text": response_text,
                    "tokens_generated": 10,
                    "tokens_per_second": 20.0,
                    "duration_ms": 500,
                }
            )
