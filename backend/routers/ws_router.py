import json
import logging
import uuid
from typing import Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.pipeline_service import PipelineSession

logger = logging.getLogger("mindease.ws")
router = APIRouter()


@router.websocket("/ws/session")
async def session_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"New session: {session_id}")

    # Create per-session pipeline
    pipeline = PipelineSession()

    async def on_pipeline_message(msg: dict):
        """Forward pipeline output messages to the WebSocket client."""
        try:
            await websocket.send_text(json.dumps(msg))
        except Exception as e:
            logger.error(f"Failed to send message to client: {e}")

    pipeline.set_message_handler(on_pipeline_message)

    # Start pipeline (loads shared models on first session)
    started = await pipeline.start()
    if not started:
        await websocket.send_text(
            json.dumps({
                "type": "error",
                "code": "PIPELINE_START_FAILED",
                "detail": "Could not start inference pipeline",
            })
        )
        await websocket.close()
        return

    # Notify client that backend is ready
    await websocket.send_text(json.dumps({"type": "status", "status": "ready"}))

    try:
        while True:
            msg = await websocket.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"]:
                # Raw PCM16 audio from microphone -- forward to pipeline VAD
                await pipeline.process_audio(msg["bytes"])

            elif "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                    ctrl_type = data.get("type", "")
                    if ctrl_type == "session_end":
                        break
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"Session {session_id} error: {e}")
    finally:
        await pipeline.stop()
        logger.info(f"Session {session_id} cleaned up")
