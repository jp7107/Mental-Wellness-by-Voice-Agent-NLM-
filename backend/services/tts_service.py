import asyncio
import base64
import logging
from typing import AsyncIterator

from config import settings

logger = logging.getLogger("mindease.tts")


class TTSService:
    """Kokoro TTS wrapper — streams audio chunks as base64."""

    def __init__(self):
        self._pipeline = None
        self._voice = "af_heart"
        self._ready = False

    def load(self) -> bool:
        model_path = settings.resolve_path(settings.kokoro_model_path)
        if not model_path.exists():
            logger.warning(f"Kokoro model not found at {model_path} — using mock TTS")
            self._ready = True
            return True

        try:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code="en-us")
            self._ready = True
            logger.info("Kokoro TTS loaded")
            return True
        except ImportError:
            logger.warning("kokoro package not installed — using mock TTS")
            self._ready = True
            return True
        except Exception as e:
            logger.error(f"TTS load error: {e}")
            self._ready = True
            return True

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw PCM audio chunks (bytes) as they are generated."""
        if not self._ready:
            return

        if self._pipeline is None:
            # Fallback to macOS native `say` command if Kokoro isn't installed
            import os
            import subprocess
            import tempfile
            
            # Create a temporary file for the WAV output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
                
            try:
                # say command produces raw PCM inside a WAV container (16-bit, 24kHz)
                subprocess.run(["say", "-o", wav_path, "--data-format=LEI16@24000", text], check=True)
                
                # Stream the PCM data, skipping the 44-byte WAV header
                with open(wav_path, "rb") as f:
                    f.read(44)  # Skip standard RIFF/WAVE header
                    while True:
                        chunk = f.read(4096)
                        if not chunk:
                            break
                        yield chunk
            except Exception as e:
                logger.error(f"say fallback TTS error: {e}")
                # Complete fallback: yield silence
                await asyncio.sleep(0.05)
                yield b"\x00\x00" * 2400
            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            return

        # Run synchronous Kokoro in thread pool so we don't block event loop
        loop = asyncio.get_event_loop()
        try:
            import numpy as np

            def _synth():
                results = []
                try:
                    for graphemes, phonemes, audio in self._pipeline(text, voice=self._voice):
                        if audio is not None:
                            # Convert float32 to int16 PCM
                            pcm = (audio * 32767).astype(np.int16)
                            results.append(pcm.tobytes())
                except Exception as e:
                    logger.error(f"TTS synthesis error: {e}")
                return results

            chunks = await loop.run_in_executor(None, _synth)
            for chunk in chunks:
                yield chunk

        except Exception as e:
            logger.error(f"TTS stream error: {e}")

    async def synthesize_to_base64(self, text: str) -> AsyncIterator[str]:
        """
        Yield base64-encoded audio chunks suitable for JSON WebSocket frames.
        """
        async for chunk in self.synthesize(text):
            yield base64.b64encode(chunk).decode("ascii")
