import os
import queue
import threading
from pathlib import Path
from typing import Dict, Generator, Optional, Union
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/fish-speech")


from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest


class FishSpeechModel:
    """
    A simplified interface for FishSpeech TTS model.
    
    Usage:
        engine = FishSpeechModel(
            llama_checkpoint_path="fishaudio/openaudio-s1-mini",
            decoder_config_name="modded_dac_vq"
        )
        
        for audio_chunk in engine.generate_speech("Hello world", {"temperature": 0.8}):
            # Process audio chunk (PCM format)
            pass
    """
    
    def __init__(
        self,
        llama_checkpoint_path: str,
        decoder_config_name: str = "modded_dac_vq",
        device: Optional[str] = None,
        precision: Optional[torch.dtype] = None,
        compile: bool = False,
        auto_download: bool = True
    ):
        """
        Initialize FishSpeech model.
        
        Args:
            llama_checkpoint_path: Path to Llama model or HuggingFace repo ID
            decoder_config_name: Name of the decoder configuration
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            precision: Model precision (torch.float16, torch.bfloat16, etc.)
            compile: Whether to compile the model for faster inference
            auto_download: Whether to automatically download models from HuggingFace
        """
        self.llama_checkpoint_path = llama_checkpoint_path
        self.decoder_config_name = decoder_config_name
        self.compile = compile
        
        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info("MPS is available, running on MPS.")
            elif torch.xpu.is_available():
                device = "xpu"
                logger.info("XPU is available, running on XPU.")
            elif torch.cuda.is_available():
                device = "cuda"
                logger.info("CUDA is available, running on CUDA.")
            else:
                device = "cpu"
                logger.info("Running on CPU.")
        
        self.device = device
        
        # Set precision
        if precision is None:
            if device == "cuda":
                precision = torch.bfloat16
            else:
                precision = torch.float32
        self.precision = precision
        
        # Download models if needed
        if auto_download:
            self._download_models()
        
        # Initialize models
        self._initialize_models()
        
        # Warm up the model
        self._warm_up()
    
    def _download_models(self):
        """Download models from HuggingFace if they don't exist locally."""
        if not os.path.exists(self.llama_checkpoint_path):
            logger.info(f"Downloading model from {self.llama_checkpoint_path}")
            
            # Check if it's a HuggingFace repo ID
            if "/" in self.llama_checkpoint_path and not os.path.exists(self.llama_checkpoint_path):
                repo_id = self.llama_checkpoint_path
                local_dir = f"/root/.cache/huggingface/hub/models--fishaudio--openaudio-s1-mini/snapshots/e735588a19774e58bdc9661d6d1e6e927cd18ef7"
                
                files_to_download = [
                    ".gitattributes",
                    "model.pth",
                    "README.md",
                    "special_tokens.json",
                    "tokenizer.tiktoken",
                    "config.json",
                    "codec.pth",
                ]
                print(os.listdir(local_dir))
                
                os.makedirs(local_dir, exist_ok=True)
                for file in files_to_download:
                    file_path = os.path.join(local_dir, file)
                    if not os.path.exists(file_path):
                        logger.info(f"Downloading {file}...")
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=file,
                            local_dir=local_dir,
                        )
                self.llama_checkpoint_path = local_dir
                self.decoder_checkpoint_path = os.path.join(local_dir, "codec.pth")
            else:
                self.decoder_checkpoint_path = os.path.join(self.llama_checkpoint_path, "codec.pth")
        else:
            self.decoder_checkpoint_path = os.path.join(self.llama_checkpoint_path, "codec.pth")
    
    def _initialize_models(self):
        """Initialize the Llama and decoder models."""
        logger.info("Loading Llama model...")
        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.llama_checkpoint_path,
            device=self.device,
            precision=self.precision,
            compile=self.compile,
        )
        
        logger.info("Loading VQ-GAN decoder model...")
        self.decoder_model = load_decoder_model(
            config_name=self.decoder_config_name,
            checkpoint_path=self.decoder_checkpoint_path,
            device=self.device,
        )
        
        logger.info("Creating inference engine...")
        self.inference_engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            compile=self.compile,
            precision=self.precision,
        )
    
    def _warm_up(self):
        """Warm up the model to avoid first-time latency."""
        logger.info("Warming up the model...")
        try:
            list(self.inference_engine.inference(
                ServeTTSRequest(
                    text="Hello world.",
                    references=[],
                    reference_id=None,
                    max_new_tokens=1024,
                    chunk_length=200,
                    top_p=0.7,
                    repetition_penalty=1.5,
                    temperature=0.7,
                    format="wav",
                )
            ))
            logger.info("Model warmed up successfully.")
        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")
    
    def generate_speech(
        self,
        text: str,
        options: Optional[Dict] = None
    ) -> Generator[tuple, None, None]:
        """
        Generate speech from text with streaming audio chunks.
        
        Args:
            text: Text to convert to speech
            options: Generation options including:
                - voice_id: Voice ID for reference
                - max_tokens: Maximum tokens to generate
                - temperature: Sampling temperature (0.1-1.0)
                - top_p: Top-p sampling parameter (0.1-1.0)
                - top_k: Top-k sampling parameter
                - min_p: Minimum probability threshold
                - pre_buffer_size: Pre-buffer size for streaming
                - chunk_length: Length of audio chunks
                - repetition_penalty: Repetition penalty (0.9-2.0)
                - streaming: Whether to stream audio
                - format: Output format ('wav', 'pcm', 'mp3')
                - seed: Random seed for reproducibility
                - normalize: Whether to normalize text
                - references: List of reference audio/text pairs
                - reference_id: Reference ID for voice cloning
        
        Yields:
            Tuple of (sample_rate, audio_chunk) where audio_chunk is numpy array
        """
        if options is None:
            options = {}
        
        # Set default options
        default_options = {
            "max_tokens": 1024,
            "temperature": 0.8,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "chunk_length": 200,
            "streaming": True,
            "format": "pcm",
            "normalize": True,
            "references": [],
            "reference_id": None,
            "seed": None,
            "use_memory_cache": "off"
        }
        
        # Update with user options
        default_options.update(options)
        
        # Create request
        request = ServeTTSRequest(
            text=text,
            max_new_tokens=default_options["max_tokens"],
            temperature=default_options["temperature"],
            top_p=default_options["top_p"],
            repetition_penalty=default_options["repetition_penalty"],
            chunk_length=default_options["chunk_length"],
            streaming=default_options["streaming"],
            format=default_options["format"],
            normalize=default_options["normalize"],
            references=default_options["references"],
            reference_id=default_options["reference_id"],
            seed=default_options["seed"],
            use_memory_cache=default_options["use_memory_cache"]
        )
        
        # Generate speech
        for result in self.inference_engine.inference(request):
            if result.code == "error":
                raise RuntimeError(f"Speech generation failed: {result.error}")
            elif result.code == "header":
                # Skip header for PCM streaming
                continue
            elif result.code == "segment":
                # Yield audio chunk
                if result.audio is not None:
                    sample_rate, audio_chunk = result.audio
                    yield (sample_rate, audio_chunk)
            elif result.code == "final":
                # Final audio chunk
                if result.audio is not None:
                    sample_rate, audio_chunk = result.audio
                    yield (sample_rate, audio_chunk)
                break
    
    def generate_speech_complete(
        self,
        text: str,
        options: Optional[Dict] = None
    ) -> tuple:
        """
        Generate complete speech from text (non-streaming).
        
        Args:
            text: Text to convert to speech
            options: Generation options (same as generate_speech)
        
        Returns:
            Tuple of (sample_rate, complete_audio) where complete_audio is numpy array
        """
        if options is None:
            options = {}
        
        # Disable streaming for complete generation
        options["streaming"] = False
        
        audio_chunks = []
        sample_rate = None
        
        for sr, chunk in self.generate_speech(text, options):
            if sample_rate is None:
                sample_rate = sr
            audio_chunks.append(chunk)
        
        if not audio_chunks:
            raise RuntimeError("No audio generated")
        
        complete_audio = np.concatenate(audio_chunks, axis=0)
        return (sample_rate, complete_audio)
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'llama_queue'):
            # Signal the worker thread to stop
            self.llama_queue.put(None)


# Convenience function for quick usage
def create_fish_speech_model(
    llama_checkpoint_path: str = "fishaudio/openaudio-s1-mini",
    decoder_config_name: str = "modded_dac_vq",
    device: Optional[str] = None,
    compile: bool = False
) -> FishSpeechModel:
    """
    Create a FishSpeech model with default settings.
    
    Args:
        llama_checkpoint_path: Path to Llama model or HuggingFace repo ID
        decoder_config_name: Name of the decoder configuration
        device: Device to run on (None for auto-detection)
        compile: Whether to compile the model for faster inference
    
    Returns:
        Initialized FishSpeechModel instance
    """
    return FishSpeechModel(
        llama_checkpoint_path=llama_checkpoint_path,
        decoder_config_name=decoder_config_name,
        device=device,
        compile=compile
    ) 