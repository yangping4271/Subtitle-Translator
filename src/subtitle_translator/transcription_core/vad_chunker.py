"""
VAD-based audio chunking module for intelligent speech segmentation.
Inspired by Qwen3-ASR-Toolkit's VAD implementation.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np

try:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    logging.warning("Silero VAD not available. Install with: pip install silero-vad torch torchaudio")

from ..logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class VADConfig:
    """Configuration for VAD-based audio chunking."""

    # VAD model parameters
    min_speech_duration_ms: int = 1500  # Minimum speech duration to keep
    min_silence_duration_ms: int = 500  # Minimum silence duration for splitting

    # Chunking parameters
    target_chunk_duration_s: int = 120  # Target chunk duration in seconds
    max_chunk_duration_s: int = 180  # Maximum chunk duration in seconds

    # Advanced parameters
    speech_pad_ms: int = 30  # Padding around speech segments
    threshold: float = 0.5  # Speech detection threshold

    # Model settings
    use_onnx: bool = True  # Use ONNX model for better performance


class VADChunker:
    """Intelligent audio chunking using Voice Activity Detection."""

    def __init__(self, config: Optional[VADConfig] = None):
        """
        Initialize VAD chunker.

        Args:
            config: VAD configuration. Uses defaults if not provided.
        """
        if not VAD_AVAILABLE:
            raise RuntimeError("Silero VAD is not installed. Please install it first.")

        self.config = config or VADConfig()
        self._vad_model = None

    @property
    def vad_model(self):
        """Lazy load VAD model."""
        if self._vad_model is None:
            logger.info("加载 Silero VAD 模型...")
            self._vad_model = load_silero_vad(onnx=self.config.use_onnx)
            logger.info("✅ VAD 模型加载完成")
        return self._vad_model

    def detect_speech_segments(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[dict]:
        """
        Detect speech segments in audio.

        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            List of speech segments with start/end timestamps
        """
        # Convert to torch tensor for VAD
        audio_tensor = torch.from_numpy(audio).float()

        # VAD parameters
        vad_params = {
            'sampling_rate': sample_rate,
            'min_speech_duration_ms': self.config.min_speech_duration_ms,
            'min_silence_duration_ms': self.config.min_silence_duration_ms,
            'speech_pad_ms': self.config.speech_pad_ms,
            'threshold': self.config.threshold,
            'return_seconds': False,  # Return sample indices
        }

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            self.vad_model,
            **vad_params
        )

        return speech_timestamps

    def find_split_points(
        self,
        audio_length_samples: int,
        speech_segments: List[dict],
        sample_rate: int
    ) -> List[int]:
        """
        Find optimal split points based on speech segments.

        Args:
            audio_length_samples: Total audio length in samples
            speech_segments: Detected speech segments
            sample_rate: Sample rate of the audio

        Returns:
            List of split points (sample indices)
        """
        # Convert durations to samples
        target_chunk_samples = self.config.target_chunk_duration_s * sample_rate
        max_chunk_samples = self.config.max_chunk_duration_s * sample_rate

        # Initialize potential split points with silence regions
        potential_splits = {0, audio_length_samples}

        # Add start of each speech segment as potential split
        for segment in speech_segments:
            potential_splits.add(segment['start'])

        # Sort split points
        sorted_splits = sorted(list(potential_splits))

        # Find optimal splits near target duration
        final_splits = {0, audio_length_samples}
        current_pos = 0

        while current_pos + target_chunk_samples < audio_length_samples:
            # Find split point closest to target position
            target_pos = current_pos + target_chunk_samples

            # Find the closest split point to target
            closest_split = min(
                sorted_splits,
                key=lambda x: abs(x - target_pos) if x > current_pos else float('inf')
            )

            # Check if chunk would be too long
            if closest_split - current_pos <= max_chunk_samples:
                final_splits.add(closest_split)
                current_pos = closest_split
            else:
                # Force split at max duration
                forced_split = current_pos + max_chunk_samples
                final_splits.add(forced_split)
                current_pos = forced_split

        return sorted(list(final_splits))

    def chunk_audio_with_vad(
        self,
        audio: mx.array,
        sample_rate: int
    ) -> List[Tuple[mx.array, Tuple[int, int]]]:
        """
        Chunk audio using VAD for intelligent segmentation.

        Args:
            audio: Audio waveform as MLX array
            sample_rate: Sample rate of the audio

        Returns:
            List of (audio_chunk, (start_sample, end_sample)) tuples
        """
        # Convert MLX array to numpy for VAD processing
        audio_np = np.array(audio)
        audio_length_samples = len(audio_np)
        audio_duration_s = audio_length_samples / sample_rate

        logger.info(f"🎵 处理音频: 时长 {audio_duration_s:.1f} 秒")

        # Short audio doesn't need chunking
        if audio_duration_s <= self.config.max_chunk_duration_s:
            logger.info("✅ 音频较短，无需分块")
            return [(audio, (0, audio_length_samples))]

        try:
            # Detect speech segments
            logger.info("🔍 检测语音活动...")
            speech_segments = self.detect_speech_segments(audio_np, sample_rate)

            if not speech_segments:
                logger.warning("⚠️ 未检测到语音活动，使用固定分块")
                return self._fallback_chunking(audio, sample_rate)

            logger.info(f"✅ 检测到 {len(speech_segments)} 个语音段")

            # Find optimal split points
            split_points = self.find_split_points(
                audio_length_samples,
                speech_segments,
                sample_rate
            )

            # Create chunks
            chunks = []
            for i in range(len(split_points) - 1):
                start_sample = split_points[i]
                end_sample = split_points[i + 1]

                # Extract chunk
                chunk = audio[start_sample:end_sample]
                chunks.append((chunk, (start_sample, end_sample)))

                # Log chunk info
                chunk_duration = (end_sample - start_sample) / sample_rate
                logger.info(
                    f"  分块 {i+1}/{len(split_points)-1}: "
                    f"{chunk_duration:.1f} 秒 "
                    f"[{start_sample/sample_rate:.1f}s - {end_sample/sample_rate:.1f}s]"
                )

            return chunks

        except Exception as e:
            logger.error(f"VAD 分块失败: {e}")
            logger.info("回退到固定分块模式")
            return self._fallback_chunking(audio, sample_rate)

    def _fallback_chunking(
        self,
        audio: mx.array,
        sample_rate: int
    ) -> List[Tuple[mx.array, Tuple[int, int]]]:
        """
        Fallback to fixed-duration chunking.

        Args:
            audio: Audio waveform as MLX array
            sample_rate: Sample rate of the audio

        Returns:
            List of (audio_chunk, (start_sample, end_sample)) tuples
        """
        max_chunk_samples = self.config.max_chunk_duration_s * sample_rate
        audio_length_samples = len(audio)

        chunks = []
        for start in range(0, audio_length_samples, max_chunk_samples):
            end = min(start + max_chunk_samples, audio_length_samples)
            chunk = audio[start:end]

            if len(chunk) > 0:
                chunks.append((chunk, (start, end)))

        logger.info(f"使用固定分块: 生成 {len(chunks)} 个块")
        return chunks


def create_vad_chunker(
    enable_vad: bool = True,
    config: Optional[VADConfig] = None
) -> Optional[VADChunker]:
    """
    Create a VAD chunker instance if enabled and available.

    Args:
        enable_vad: Whether to enable VAD chunking
        config: VAD configuration

    Returns:
        VADChunker instance or None if disabled/unavailable
    """
    if not enable_vad:
        logger.info("VAD 分块已禁用")
        return None

    if not VAD_AVAILABLE:
        logger.warning("VAD 不可用，使用传统分块")
        return None

    try:
        return VADChunker(config)
    except Exception as e:
        logger.error(f"创建 VAD 分块器失败: {e}")
        return None