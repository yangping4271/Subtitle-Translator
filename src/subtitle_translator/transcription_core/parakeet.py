import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional
import platform
import re

from ..logger import setup_logger

import mlx.core as mx
import mlx.nn as nn

from . import tokenizer
from .alignment import (
    AlignedResult,
    AlignedToken,
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    sentences_to_result,
    tokens_to_sentences,
)
from .audio import PreprocessArgs, get_logmel, load_audio
from .cache import ConformerCache, RotatingConformerCache
from .conformer import Conformer, ConformerArgs
from .ctc import AuxCTCArgs, ConvASRDecoder, ConvASRDecoderArgs
from .rnnt import JointArgs, JointNetwork, PredictArgs, PredictNetwork
from .vad_chunker import VADChunker, VADConfig, create_vad_chunker


def get_optimal_chunk_duration(audio_duration_seconds: float, logger=None) -> Optional[float]:
    """
    根据系统性能和音频长度智能选择最佳分块时长
    
    Args:
        audio_duration_seconds: 音频总时长（秒）
        logger: 日志记录器
    
    Returns:
        chunk_duration: 分块时长（秒），None表示不分块
    """
    if logger is None:
        logger = setup_logger(__name__)
    
    # 1. macOS内存检测策略
    memory_gb = None
    
    # 策略1: 使用psutil（最准确）
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        logger.debug(f"psutil检测到系统内存: {memory_gb:.1f}GB")
    except ImportError:
        logger.debug("psutil不可用，尝试macOS原生方法")
    except Exception as e:
        logger.debug(f"psutil检测失败: {e}")
    
    # 策略2: macOS原生sysctl（高效可靠）
    if memory_gb is None:
        try:
            import subprocess
            result = subprocess.run(['sysctl', 'hw.memsize'], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                mem_bytes = int(result.stdout.split()[-1])
                memory_gb = mem_bytes / (1024**3)
                logger.debug(f"macOS sysctl检测到系统内存: {memory_gb:.1f}GB")
        except Exception as e:
            logger.debug(f"macOS内存检测失败: {e}")
    
    # 保险策略：合理默认值
    if memory_gb is None:
        memory_gb = 16  # macOS设备通常至少16GB
        logger.warning("⚠️  内存检测失败，使用默认值16GB")
    else:
        logger.info(f"✅ 检测到系统内存: {memory_gb:.1f}GB")
    
    # 2. Apple Silicon检测（macOS优化）
    is_apple_silicon = False
    chip_info = "Intel Mac"
    
    # 策略1: 使用platform.machine()检测架构（最快最准确）
    try:
        machine = platform.machine()
        if machine == 'arm64':
            is_apple_silicon = True
            chip_info = "Apple Silicon"
            logger.debug(f"检测到ARM64架构: {machine}")
    except Exception as e:
        logger.debug(f"架构检测失败: {e}")
    
    # 策略2: 获取详细芯片信息（sysctl更快更可靠）
    if is_apple_silicon:
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                brand_string = result.stdout.strip()
                if 'Apple' in brand_string:
                    chip_info = brand_string
                    logger.debug(f"获取到详细芯片信息: {brand_string}")
        except Exception as e:
            logger.debug(f"芯片信息获取失败: {e}")
    
    logger.info(f"💻 系统配置: {memory_gb:.1f}GB内存, {chip_info}")
    
    # 3. 基于音频长度的基础策略
    if audio_duration_seconds <= 8 * 60:  # 8分钟内
        logger.info("🎯 音频较短，使用单次处理")
        return None
    
    # 4. 根据系统性能调整分块策略
    if is_apple_silicon:
        # Apple Silicon优化策略
        if memory_gb >= 32:
            # 32GB+ 内存：大块处理
            chunk_duration = min(20 * 60, audio_duration_seconds * 0.6)  # 最大20分钟或音频60%
            strategy = "高性能"
        elif memory_gb >= 16:
            # 16GB 内存：平衡策略
            chunk_duration = min(12 * 60, audio_duration_seconds * 0.5)  # 最大12分钟或音频50%
            strategy = "平衡"
        elif memory_gb >= 8:
            # 8GB 内存：保守策略
            chunk_duration = min(8 * 60, audio_duration_seconds * 0.4)   # 最大8分钟或音频40%
            strategy = "保守"
        else:
            # < 8GB 内存：超保守策略
            chunk_duration = min(5 * 60, audio_duration_seconds * 0.3)   # 最大5分钟或音频30%
            strategy = "超保守"
    else:
        # Intel/其他平台策略（更保守）
        if memory_gb >= 32:
            chunk_duration = min(15 * 60, audio_duration_seconds * 0.5)
            strategy = "Intel高性能"
        elif memory_gb >= 16:
            chunk_duration = min(10 * 60, audio_duration_seconds * 0.4)
            strategy = "Intel平衡"
        elif memory_gb >= 8:
            chunk_duration = min(6 * 60, audio_duration_seconds * 0.3)
            strategy = "Intel保守"
        else:
            chunk_duration = min(4 * 60, audio_duration_seconds * 0.25)
            strategy = "Intel超保守"
    
    # 5. 确保最小分块不少于2分钟（避免过度分块）
    if chunk_duration < 2 * 60:
        chunk_duration = 2 * 60
    
    logger.info(f"📦 {strategy}策略: {chunk_duration/60:.1f}分钟分块")
    return chunk_duration


@dataclass
class TDTDecodingArgs:
    model_type: str
    durations: list[int]
    greedy: dict | None


@dataclass
class RNNTDecodingArgs:
    greedy: dict | None


@dataclass
class CTCDecodingArgs:
    greedy: dict | None


@dataclass
class ParakeetTDTArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: PredictArgs
    joint: JointArgs
    decoding: TDTDecodingArgs


@dataclass
class ParakeetRNNTArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: PredictArgs
    joint: JointArgs
    decoding: RNNTDecodingArgs


@dataclass
class ParakeetCTCArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: ConvASRDecoderArgs
    decoding: CTCDecodingArgs


@dataclass
class ParakeetTDTCTCArgs(ParakeetTDTArgs):
    aux_ctc: AuxCTCArgs


# API
@dataclass
class DecodingConfig:
    decoding: str = "greedy"


# common methods
class BaseParakeet(nn.Module):
    """Base parakeet model for interface purpose"""

    def __init__(self, preprocess_args: PreprocessArgs, encoder_args: ConformerArgs):
        super().__init__()

        self.preprocessor_config = preprocess_args
        self.encoder_config = encoder_args

        self.encoder = Conformer(encoder_args)

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        """
        Generate transcription results from the Parakeet model, handling batches and single input.
        Args:
            mel (mx.array):
                Mel-spectrogram input with shape [batch, sequence, mel_dim] for
                batch processing or [sequence, mel_dim] for single input.
            decoding_config (DecodingConfig, optional):
                Configuration object that controls decoding behavior and
                parameters for the generation process. Defaults to DecodingConfig().
        Returns:
            list[AlignedResult]: List of transcription results with aligned tokens
                and sentences, one for each input in the batch.
        """
        raise NotImplementedError

    def transcribe(
        self,
        path: Path | str,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        chunk_duration: Optional[float] = None,
        overlap_duration: float = 15.0,
        chunk_callback: Optional[Callable] = None,
        use_vad: bool = True,
    ) -> AlignedResult:
        """
        Transcribe an audio file, with optional chunking for long files.
        Args:
            path (Path | str):
                Path to the audio file to be transcribed.
            dtype (mx.Dtype, optional):
                Data type for processing the audio. Defaults to mx.bfloat16.
            chunk_duration (float, optional):
                If provided, splits audio into chunks of this length (in seconds)
                for processing. When None, processes the entire file at once.
                Defaults to None.
            overlap_duration (float, optional):
                Overlap between consecutive chunks in seconds. Only used when
                chunk_duration is specified. Defaults to 15.0.
            chunk_callback (Callable, optional):
                A function to call when each chunk is processed. The callback
                is called with (current_position, total_position) arguments
                to track progress. Defaults to None.
            use_vad (bool, optional):
                Whether to use VAD-based intelligent chunking. When True, uses
                Voice Activity Detection to find optimal split points at silence.
                Defaults to True.
        Returns:
            AlignedResult: Transcription result with aligned tokens and sentences.
        """
        audio_path = Path(path)
        audio_data = load_audio(audio_path, self.preprocessor_config.sample_rate, dtype)

        # 添加基础日志记录
        logger = setup_logger(__name__)

        # Check if we should use VAD chunking
        # VAD chunking is only used when chunk_duration is negative (smart chunking)
        # If user explicitly sets a positive chunk_duration, respect their choice
        if use_vad and chunk_duration is not None and chunk_duration < 0:
            logger.info("🎯 启用 VAD 智能分块")
            vad_chunker = create_vad_chunker(enable_vad=True)

            if vad_chunker is not None:
                # Use VAD-based chunking
                chunks = vad_chunker.chunk_audio_with_vad(
                    audio_data,
                    self.preprocessor_config.sample_rate
                )

                if len(chunks) == 1:
                    # Single chunk, process directly
                    mel = get_logmel(chunks[0][0], self.preprocessor_config)
                    return self.generate(mel)[0]

                # Process multiple chunks
                all_tokens = []
                for i, (chunk_audio, (start_sample, end_sample)) in enumerate(chunks):
                    if chunk_callback is not None:
                        chunk_callback(end_sample, len(audio_data))

                    # Process chunk
                    chunk_mel = get_logmel(chunk_audio, self.preprocessor_config)
                    chunk_result = self.generate(chunk_mel)[0]

                    # Adjust timestamps
                    chunk_offset = start_sample / self.preprocessor_config.sample_rate
                    for sentence in chunk_result.sentences:
                        for token in sentence.tokens:
                            token.start += chunk_offset
                            token.end = token.start + token.duration

                    all_tokens.extend(chunk_result.tokens)

                # Merge results - 由于 VAD 分块没有重叠，直接使用 tokens
                merged_sentences = tokens_to_sentences(all_tokens)
                return sentences_to_result(merged_sentences)
            else:
                logger.warning("VAD 不可用，回退到传统分块")
                # Fall back to regular chunking

        if chunk_duration is None:
            logger.info("✅ 使用单次处理（无分块）")
            mel = get_logmel(audio_data, self.preprocessor_config)
            return self.generate(mel)[0]

        audio_length_seconds = len(audio_data) / self.preprocessor_config.sample_rate
        logger.info(f"🎵 音频时长: {audio_length_seconds/60:.1f}分钟")

        # 智能分块策略：如果chunk_duration为负数，启用自动优化
        if chunk_duration < 0:
            chunk_duration = get_optimal_chunk_duration(audio_length_seconds, logger)
            if chunk_duration is None:
                mel = get_logmel(audio_data, self.preprocessor_config)
                return self.generate(mel)[0]
        elif audio_length_seconds <= chunk_duration:
            logger.info("✅ 音频时长小于分块时长，使用单次处理")
            mel = get_logmel(audio_data, self.preprocessor_config)
            return self.generate(mel)[0]

        chunk_samples = int(chunk_duration * self.preprocessor_config.sample_rate)
        overlap_samples = int(overlap_duration * self.preprocessor_config.sample_rate)
        
        total_chunks = (len(audio_data) + chunk_samples - overlap_samples - 1) // (chunk_samples - overlap_samples)
        logger.info(f"🔧 实际分块: {total_chunks}块，每块{chunk_duration/60:.1f}分钟，重叠{overlap_duration}秒")

        all_tokens = []

        for start in range(0, len(audio_data), chunk_samples - overlap_samples):
            end = min(start + chunk_samples, len(audio_data))

            if chunk_callback is not None:
                chunk_callback(end, len(audio_data))

            # 检查是否为最后一个chunk，如果是且有内容则不跳过
            is_last_chunk = (start + chunk_samples >= len(audio_data))
            if end - start < self.preprocessor_config.hop_length:
                if not is_last_chunk or end <= start:
                    break  # prevent zero-length log mel
                # 最后一个chunk即使很短也要处理，避免丢失内容

            chunk_audio = audio_data[start:end]
            chunk_mel = get_logmel(chunk_audio, self.preprocessor_config)

            chunk_result = self.generate(chunk_mel)[0]

            chunk_offset = start / self.preprocessor_config.sample_rate
            for sentence in chunk_result.sentences:
                for token in sentence.tokens:
                    token.start += chunk_offset
                    token.end = token.start + token.duration

            if all_tokens:
                try:
                    all_tokens = merge_longest_contiguous(
                        all_tokens,
                        chunk_result.tokens,
                        overlap_duration=overlap_duration,
                    )
                    logger.debug(f"✅ 严格合并成功：精确匹配重叠区域")
                except RuntimeError as e:
                    logger.warning(f"🔄 严格合并未达标，启用智能合并：{e}")
                    try:
                        before_count = len(all_tokens)
                        all_tokens = merge_longest_common_subsequence(
                            all_tokens,
                            chunk_result.tokens,
                            overlap_duration=overlap_duration,
                        )
                        after_count = len(all_tokens)
                        added_tokens = after_count - before_count
                        logger.info(f"✅ 智能合并完成：成功添加{added_tokens}个新token，无内容丢失")
                    except Exception as e2:
                        logger.error(f"❌ 所有合并算法都失败：{e2}")
                        # 保险合并：简单拼接
                        before_count = len(all_tokens)
                        all_tokens.extend(chunk_result.tokens)
                        added_tokens = len(chunk_result.tokens)
                        logger.warning(f"🆘 使用保险合并：直接添加{added_tokens}个token（可能有重复）")
            else:
                all_tokens = chunk_result.tokens

        result = sentences_to_result(tokens_to_sentences(all_tokens))
        logger.info(f"🎯 转录完成: {len(all_tokens)}个token，{len(result.sentences)}个句子")
        return result

    def transcribe_stream(
        self,
        context_size: tuple[int, int] = (256, 256),
        depth=1,
        *,
        keep_original_attention: bool = False,
        decoding_config: DecodingConfig = DecodingConfig(),
    ) -> "StreamingParakeet":
        """
        Create a StreamingParakeet object for real-time (streaming) inference.
        Args:
            context_size (tuple[int, int], optional):
                A pair (left_context, right_context) for attention context windows.
            depth (int, optional):
                How many encoder layers will carry over their key/value
                cache (i.e. hidden state) exactly across chunks. Because
                we use local (non-causal) attention, the cache is only
                guaranteed to match a full forward pass up through each
                cached layer:
                    • depth=1 (default): only the first encoder layer's
                    cache matches exactly.
                    • depth=2: the first two layers match, and so on.
                    • depth=N (model's total layers): full equivalence to
                    a non-streaming forward pass.
                Setting `depth` larger than the model's total number
                of encoder layers won't have any impacts.
            keep_original_attention (bool, optional):
                Whether to preserve the original attention class
                during streaming inference. Defaults to False. (Will switch to local attention.)
            decoding_config (DecodingConfig, optional):
                Configuration object that controls decoding behavior
                Defaults to DecodingConfig().
        Returns:
            StreamingParakeet: A context manager for streaming inference.
        """
        return StreamingParakeet(
            self,
            context_size,
            depth,
            decoding_config=decoding_config,
            keep_original_attention=keep_original_attention,
        )


# models
class ParakeetTDT(BaseParakeet):
    """MLX Implementation of Parakeet-TDT Model"""

    def __init__(self, args: ParakeetTDTArgs):
        super().__init__(args.preprocessor, args.encoder)

        assert args.decoding.model_type == "tdt", "Model must be a TDT model"

        self.vocabulary = args.joint.vocabulary
        self.durations = args.decoding.durations
        self.max_symbols: int | None = (
            args.decoding.greedy.get("max_symbols", None)
            if args.decoding.greedy
            else None
        )

        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def decode(
        self,
        features: mx.array,
        lengths: Optional[mx.array] = None,
        last_token: Optional[list[Optional[int]]] = None,
        hidden_state: Optional[list[Optional[tuple[mx.array, mx.array]]]] = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> tuple[list[list[AlignedToken]], list[Optional[tuple[mx.array, mx.array]]]]:
        """Run TDT decoder with features, optional length and decoder state. Outputs list[list[AlignedToken]] and updated hidden state"""
        assert config.decoding == "greedy", (
            "Only greedy decoding is supported for TDT decoder now"
        )

        B, S, *_ = features.shape

        if hidden_state is None:
            hidden_state = list([None] * B)

        if lengths is None:
            lengths = mx.array([S] * B)

        if last_token is None:
            last_token = list([None] * B)

        results = []
        for batch in range(B):
            hypothesis = []

            feature = features[batch : batch + 1]
            length = int(lengths[batch])

            step = 0
            new_symbols = 0

            while step < length:
                # decoder pass
                decoder_out, (hidden, cell) = self.decoder(
                    mx.array([[last_token[batch]]])
                    if last_token[batch] is not None
                    else None,
                    hidden_state[batch],
                )
                decoder_out = decoder_out.astype(feature.dtype)
                decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )

                # joint pass
                joint_out = self.joint(feature[:, step : step + 1], decoder_out)

                # sampling
                pred_token = int(
                    mx.argmax(joint_out[0, 0, :, : len(self.vocabulary) + 1])
                )
                decision = int(
                    mx.argmax(joint_out[0, 0, :, len(self.vocabulary) + 1 :])
                )

                # tdt decoding rule
                if pred_token != len(self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=step
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            duration=self.durations[decision]
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            text=tokenizer.decode([pred_token], self.vocabulary),
                        )
                    )
                    last_token[batch] = pred_token
                    hidden_state[batch] = decoder_hidden

                step += self.durations[int(decision)]

                # prevent stucking rule
                new_symbols += 1

                if self.durations[int(decision)] != 0:
                    new_symbols = 0
                else:
                    if self.max_symbols is not None and self.max_symbols <= new_symbols:
                        step += 1
                        new_symbols = 0

            results.append(hypothesis)

        return results, hidden_state

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)

        result, _ = self.decode(features, lengths, config=decoding_config)

        return [
            sentences_to_result(tokens_to_sentences(hypothesis))
            for hypothesis in result
        ]


class ParakeetRNNT(BaseParakeet):
    """MLX Implementation of Parakeet-RNNT Model"""

    def __init__(self, args: ParakeetRNNTArgs):
        super().__init__(args.preprocessor, args.encoder)

        self.vocabulary = args.joint.vocabulary
        self.max_symbols: int | None = (
            args.decoding.greedy.get("max_symbols", None)
            if args.decoding.greedy
            else None
        )

        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def decode(
        self,
        features: mx.array,
        lengths: Optional[mx.array] = None,
        last_token: Optional[list[Optional[int]]] = None,
        hidden_state: Optional[list[Optional[tuple[mx.array, mx.array]]]] = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> tuple[list[list[AlignedToken]], list[Optional[tuple[mx.array, mx.array]]]]:
        """Run TDT decoder with features, optional length and decoder state. Outputs list[list[AlignedToken]] and updated hidden state"""
        assert config.decoding == "greedy", (
            "Only greedy decoding is supported for RNNT decoder now"
        )

        B, S, *_ = features.shape

        if hidden_state is None:
            hidden_state = list([None] * B)

        if lengths is None:
            lengths = mx.array([S] * B)

        if last_token is None:
            last_token = list([None] * B)

        results = []
        for batch in range(B):
            hypothesis = []

            feature = features[batch : batch + 1]
            length = int(lengths[batch])

            step = 0
            new_symbols = 0

            while step < length:
                # decoder pass
                decoder_out, (hidden, cell) = self.decoder(
                    mx.array([[last_token[batch]]])
                    if last_token[batch] is not None
                    else None,
                    hidden_state[batch],
                )
                decoder_out = decoder_out.astype(feature.dtype)
                decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )

                # joint pass
                joint_out = self.joint(feature[:, step : step + 1], decoder_out)

                # sampling
                pred_token = int(mx.argmax(joint_out[0, 0]))

                # rnnt decoding rule
                if pred_token != len(self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=step
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            duration=1
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            text=tokenizer.decode([pred_token], self.vocabulary),
                        )
                    )
                    last_token[batch] = pred_token
                    hidden_state[batch] = decoder_hidden

                    # prevent stucking
                    new_symbols += 1
                    if self.max_symbols is not None and self.max_symbols <= new_symbols:
                        step += 1
                        new_symbols = 0
                else:
                    step += 1
                    new_symbols = 0

            results.append(hypothesis)

        return results, hidden_state

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)

        result, _ = self.decode(features, lengths, config=decoding_config)

        return [
            sentences_to_result(tokens_to_sentences(hypothesis))
            for hypothesis in result
        ]


class ParakeetCTC(BaseParakeet):
    """MLX Implementation of Parakeet-CTC Model"""

    def __init__(self, args: ParakeetCTCArgs):
        super().__init__(args.preprocessor, args.encoder)

        self.vocabulary = args.decoder.vocabulary

        self.decoder = ConvASRDecoder(args.decoder)

    def decode(
        self,
        features: mx.array,
        lengths: mx.array,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> list[list[AlignedToken]]:
        """Run CTC decoder with features and lengths. Outputs list[list[AlignedToken]]."""
        B, S, *_ = features.shape

        logits = self.decoder(features)
        mx.eval(logits, lengths)

        results = []
        for batch in range(B):
            length = int(lengths[batch])
            predictions = logits[batch, :length]
            best_tokens = mx.argmax(predictions, axis=1)

            hypothesis = []
            token_boundaries = []
            prev_token = -1

            for t, token_id in enumerate(best_tokens):
                token_idx = int(token_id)

                if token_idx == len(self.vocabulary):
                    continue

                if token_idx == prev_token:
                    continue

                if prev_token != -1:
                    token_start_time = (
                        token_boundaries[-1][0]
                        * self.encoder_config.subsampling_factor
                        / self.preprocessor_config.sample_rate
                        * self.preprocessor_config.hop_length
                    )

                    token_end_time = (
                        t
                        * self.encoder_config.subsampling_factor
                        / self.preprocessor_config.sample_rate
                        * self.preprocessor_config.hop_length
                    )

                    token_duration = token_end_time - token_start_time

                    hypothesis.append(
                        AlignedToken(
                            prev_token,
                            start=token_start_time,
                            duration=token_duration,
                            text=tokenizer.decode([prev_token], self.vocabulary),
                        )
                    )

                token_boundaries.append((t, None))
                prev_token = token_idx

            if prev_token != -1:
                last_non_blank = length - 1
                for t in range(length - 1, token_boundaries[-1][0], -1):
                    if int(best_tokens[t]) != len(self.vocabulary):
                        last_non_blank = t
                        break

                token_start_time = (
                    token_boundaries[-1][0]
                    * self.encoder_config.subsampling_factor
                    / self.preprocessor_config.sample_rate
                    * self.preprocessor_config.hop_length
                )

                token_end_time = (
                    (last_non_blank + 1)
                    * self.encoder_config.subsampling_factor
                    / self.preprocessor_config.sample_rate
                    * self.preprocessor_config.hop_length
                )

                token_duration = token_end_time - token_start_time

                hypothesis.append(
                    AlignedToken(
                        prev_token,
                        start=token_start_time,
                        duration=token_duration,
                        text=tokenizer.decode([prev_token], self.vocabulary),
                    )
                )

            results.append(hypothesis)

        return results

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)

        result = self.decode(features, lengths, config=decoding_config)

        return [
            sentences_to_result(tokens_to_sentences(hypothesis))
            for hypothesis in result
        ]


class ParakeetTDTCTC(ParakeetTDT):
    """MLX Implementation of Parakeet-TDT-CTC Model

    Has ConvASRDecoder decoder in `.ctc_decoder` but `.generate` uses TDT decoder all the times (Please open an issue if you need CTC decoder use-case!)"""

    def __init__(self, args: ParakeetTDTCTCArgs):
        super().__init__(args)

        self.ctc_decoder = ConvASRDecoder(args.aux_ctc.decoder)


# streaming
class StreamingParakeet:
    model: "BaseParakeet"
    cache: List[ConformerCache]

    audio_buffer: mx.array
    mel_buffer: Optional[mx.array]
    decoder_hidden: Optional[tuple[mx.array, mx.array]] = None
    last_token: Optional[int] = None

    finalized_tokens: list[AlignedToken]
    draft_tokens: list[AlignedToken]

    context_size: tuple[int, int]
    depth: int
    decoding_config: DecodingConfig
    keep_original_attention: bool = False

    def __init__(
        self,
        model: "BaseParakeet",
        context_size: tuple[int, int],
        depth: int = 1,
        *,
        keep_original_attention: bool = False,
        decoding_config: DecodingConfig = DecodingConfig(),
    ) -> None:
        self.context_size = context_size
        self.depth = depth
        self.decoding_config = decoding_config
        self.keep_original_attention = keep_original_attention

        self.model = model
        self.cache = [
            RotatingConformerCache(self.keep_size, cache_drop_size=self.drop_size)
            for _ in range(len(model.encoder.layers))
        ]

        self.audio_buffer = mx.array([])
        self.mel_buffer = None
        self.finalized_tokens = []
        self.draft_tokens = []

    def __enter__(self):
        if not self.keep_original_attention:
            self.model.encoder.set_attention_model(
                "rel_pos_local_attn", self.context_size
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.keep_original_attention:
            self.model.encoder.set_attention_model(
                "rel_pos"
            )  # hard-coded; might cache if there's actually new varient than rel_pos
        del self.audio_buffer
        del self.cache

        mx.clear_cache()

    @property
    def keep_size(self):
        """Indicates how many encoded feature frames to keep in KV cache"""
        return self.context_size[0]

    @property
    def drop_size(self):
        """Indicates how many encoded feature frames to drop"""
        return self.context_size[1] * self.depth

    @property
    def result(self) -> AlignedResult:
        """Transcription result"""
        return sentences_to_result(
            tokens_to_sentences(self.finalized_tokens + self.draft_tokens)
        )

    def add_audio(self, audio: mx.array) -> None:
        """Takes portion of audio and transcribe it.

        `audio` must be 1D array"""

        self.audio_buffer = mx.concat(
            [
                self.audio_buffer,
                audio,
            ],
            axis=0,
        )
        mel = get_logmel(
            self.audio_buffer[
                : (
                    len(self.audio_buffer)
                    // self.model.preprocessor_config.hop_length
                    * self.model.preprocessor_config.hop_length
                )
            ],
            self.model.preprocessor_config,
        )

        if self.mel_buffer is None:  # init
            self.mel_buffer = mel
        else:
            self.mel_buffer = mx.concat([self.mel_buffer, mel], axis=1)

        self.audio_buffer = self.audio_buffer[
            (mel.shape[1] * self.model.preprocessor_config.hop_length) :
        ]

        features, lengths = self.model.encoder(
            self.mel_buffer[
                :,
                : (
                    self.mel_buffer.shape[1]
                    // self.model.encoder_config.subsampling_factor
                    * self.model.encoder_config.subsampling_factor
                ),
            ],
            cache=self.cache,
        )
        mx.eval(features, lengths)
        length = int(lengths[0])

        # cache will automatically dropped in cache level
        leftover = self.mel_buffer.shape[1] - (
            length * self.model.encoder_config.subsampling_factor
        )
        self.mel_buffer = self.mel_buffer[
            :,
            -(
                self.drop_size * self.model.encoder_config.subsampling_factor + leftover
            ) :,
        ]

        # we decode in two phase
        # first phase: finalized region decode
        # second phase: draft region decode (will be dropped)
        finalized_length = max(0, length - self.drop_size)

        if isinstance(self.model, ParakeetTDT) or isinstance(self.model, ParakeetRNNT):
            finalized_tokens, finalized_state = self.model.decode(
                features,
                mx.array([finalized_length]),
                [self.last_token],
                [self.decoder_hidden],
                config=self.decoding_config,
            )

            self.decoder_hidden = finalized_state[0]
            self.last_token = (
                finalized_tokens[0][-1].id if len(finalized_tokens[0]) > 0 else None
            )

            draft_tokens, _ = self.model.decode(
                features[:, finalized_length:],
                mx.array(
                    [
                        features[:, finalized_length:].shape[1]
                    ]  # i believe in lazy evaluation
                ),
                [self.last_token],
                [self.decoder_hidden],
                config=self.decoding_config,
            )

            self.finalized_tokens.extend(finalized_tokens[0])
            self.draft_tokens = draft_tokens[0]
        elif isinstance(self.model, ParakeetCTC):
            finalized_tokens = self.model.decode(
                features, mx.array([finalized_length]), config=self.decoding_config
            )

            draft_tokens = self.model.decode(
                features[:, finalized_length:],
                mx.array(
                    [
                        features[:, finalized_length:].shape[1]
                    ]  # i believe in lazy evaluation
                ),
                config=self.decoding_config,
            )

            self.finalized_tokens.extend(finalized_tokens[0])
            self.draft_tokens = draft_tokens[0]
        else:
            raise NotImplementedError("This model does not support real-time decoding")
