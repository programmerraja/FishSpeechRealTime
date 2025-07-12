# type: ignore
import asyncio
import websockets
import json
import logging
import time
import os
from typing import AsyncGenerator, Optional
from index import FishSpeechModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrpheusCPUConfigs:
    def __init__(
        self,
        voice_id="tara",
        max_tokens=2048,
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        min_p=0.05,
        pre_buffer_size=0.0
    ):
        self.voice_id = voice_id
        self.max_tokens = max(100, min(5000, max_tokens))
        self.temperature = max(0.1, min(2.0, temperature))
        self.top_p = max(0.1, min(1.0, top_p))
        self.top_k = max(1, min(100, top_k))
        self.min_p = max(0.01, min(1.0, min_p))
        self.pre_buffer_size = max(0.0, min(5.0, pre_buffer_size))

# === GLOBALS ===
engine: Optional[FishSpeechModel] = None
default_config: Optional[OrpheusCPUConfigs] = None


async def load_model():
    global engine
    logger.info("Loading FishSpeech TTS model...")

    try:
        # Initialize OrpheusCpp with CPU optimizations
        engine = FishSpeechModel(
            llama_checkpoint_path="fishaudio/openaudio-s1-mini",
            # decoder_config_name="modded_dac_vq"
        )
        logger.info("FishSpeech TTS model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load FishSpeech model: {e}")
        raise


async def run_tts(text: str, tts_cfg: OrpheusCPUConfigs) -> AsyncGenerator[bytes, None]:
    if engine is None:
        raise RuntimeError("Engine not loaded")

    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    options = {
        "voice_id": tts_cfg.voice_id,
        "max_tokens": tts_cfg.max_tokens,
        "temperature": tts_cfg.temperature,
        "top_p": tts_cfg.top_p,
        "top_k": tts_cfg.top_k,
        "min_p": tts_cfg.min_p,
        "pre_buffer_size": tts_cfg.pre_buffer_size
    }

    def generate_stream():
        try:
            for sr, chunk in engine.generate_speech(text, options=options):
                if hasattr(chunk, "tobytes"):
                    chunk = chunk.tobytes()
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel for done

    # Run the blocking generator in a thread
    loop.run_in_executor(None, generate_stream)

    # Async consumer: yield each chunk as it appears
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk

async def handler(websocket):
    global default_config
    
    logger.info("New WebSocket connection.")
    async for message in websocket:
        try:
            msg = json.loads(message)
            cfg = default_config

            if "config" in msg:
                cfg = OrpheusCPUConfigs(**msg["config"])
                logger.info("Overriding config from WS.")
            
            if "text" in msg:
                text = msg.get("text", "").strip()
                if not text:
                    continue

                logger.info(f"Generating speech for text: {text[:50]}...")
                async for audio_chunk in run_tts(text, cfg):
                    await websocket.send(audio_chunk)
                logger.info("Speech generation completed.")

        except Exception as e:
            logger.error(f"Handler error: {e}")
            await websocket.send(f"ERROR: {str(e)}".encode())


async def main():
    global default_config

    await load_model()

    default_config = OrpheusCPUConfigs(
        voice_id="tara",
        max_tokens=2048,
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        min_p=0.05,
        pre_buffer_size=0.0
    )

    port = int(os.environ.get("PORT", 9802))
    async with websockets.serve(handler, "0.0.0.0", port,ping_timeout=None,ping_interval=None,compression=None):
        logger.info(f"FishSpeech TTS WebSocket server running on ws://0.0.0.0:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())