import json
import os
from pathlib import Path
from typing import Union

import mlx.core as mx
from dacite import from_dict
from mlx.utils import tree_flatten, tree_unflatten

from .parakeet import (
    BaseParakeet,
    ParakeetCTC,
    ParakeetCTCArgs,
    ParakeetRNNT,
    ParakeetRNNTArgs,
    ParakeetTDT,
    ParakeetTDTArgs,
    ParakeetTDTCTC,
    ParakeetTDTCTCArgs,
)
from .model_cache import load_cached_model
from ..logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

# 默认转录模型
DEFAULT_TRANSCRIPTION_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"


def _find_model_in_hf_cache(model_id: str) -> Path:
    """在 HuggingFace 缓存中查找模型

    Args:
        model_id: 模型 ID，格式如 "mlx-community/parakeet-tdt-0.6b-v2"

    Returns:
        模型路径，如果找到的话

    Raises:
        FileNotFoundError: 未找到模型
    """
    # HF 缓存目录
    hf_cache = os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"

    # 将模型 ID 转换为缓存目录名
    # mlx-community/parakeet-tdt-0.6b-v2 -> models--mlx-community--parakeet-tdt-0.6b-v2
    cache_model_name = model_id.replace("/", "--")
    model_cache_dir = Path(hf_cache) / "hub" / f"models--{cache_model_name}"

    if not model_cache_dir.exists():
        raise FileNotFoundError(f"未找到模型缓存: {model_cache_dir}")

    # 查找 snapshots 目录下的最新版本
    snapshots_dir = model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"未找到 snapshots 目录: {snapshots_dir}")

    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        raise FileNotFoundError(f"snapshots 目录为空: {snapshots_dir}")

    # 返回最新的 snapshot（按修改时间）
    latest_snapshot = max(snapshot_dirs, key=lambda d: d.stat().st_mtime)
    logger.info(f"从 HF 缓存加载模型: {latest_snapshot}")
    return latest_snapshot


def _get_model_path(model_path: str | None = None) -> Path:
    """获取模型路径

    优先级：
    1. 命令行参数指定的路径
    2. 配置文件中的 TRANSCRIPTION_MODEL_PATH
    3. HuggingFace 缓存中的默认模型

    Args:
        model_path: 用户指定的模型路径（可选）

    Returns:
        模型路径
    """
    # 1. 命令行参数优先
    if model_path:
        return Path(model_path)

    # 2. 配置文件中的路径
    config_path = os.environ.get("TRANSCRIPTION_MODEL_PATH")
    if config_path:
        logger.info(f"使用配置文件中的模型路径: {config_path}")
        return Path(config_path)

    # 3. 从 HF 缓存查找默认模型
    try:
        return _find_model_in_hf_cache(DEFAULT_TRANSCRIPTION_MODEL)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"未找到转录模型\n\n"
            f"默认模型: {DEFAULT_TRANSCRIPTION_MODEL}\n\n"
            f"请先下载模型：\n"
            f"  hf download {DEFAULT_TRANSCRIPTION_MODEL}\n\n"
            f"或在配置文件 (~/.config/subtitle-translator/.env) 中指定自定义路径：\n"
            f"  TRANSCRIPTION_MODEL_PATH=/path/to/model\n"
        ) from e


def from_config(config: dict) -> BaseParakeet:
    """从配置构建模型（随机权重）"""
    if (
        config.get("target")
        == "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
        and config.get("model_defaults", {}).get("tdt_durations") is not None
    ):
        cfg = from_dict(ParakeetTDTArgs, config)
        model = ParakeetTDT(cfg)
    elif (
        config.get("target")
        == "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel"
        and config.get("model_defaults", {}).get("tdt_durations") is not None
    ):
        cfg = from_dict(ParakeetTDTCTCArgs, config)
        model = ParakeetTDTCTC(cfg)
    elif (
        config.get("target")
        == "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
        and config.get("model_defaults", {}).get("tdt_durations") is None
    ):
        cfg = from_dict(ParakeetRNNTArgs, config)
        model = ParakeetRNNT(cfg)
    elif (
        config.get("target")
        == "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE"
    ):
        cfg = from_dict(ParakeetCTCArgs, config)
        model = ParakeetCTC(cfg)
    else:
        raise ValueError("不支持的模型类型")

    model.eval()  # 防止 layernorm 在推理时计算错误

    return model


def from_pretrained(
    model_path: str | None = None,
    *,
    dtype: mx.Dtype = mx.bfloat16,
    use_cache: bool = True,
    return_cache_info: bool = False
) -> Union[BaseParakeet, tuple[BaseParakeet, bool]]:
    """
    从本地路径加载模型

    Args:
        model_path: 模型本地路径（可选）
                   - None: 使用配置文件或默认模型
                   - 路径: 加载指定路径的模型
        dtype: 模型数据类型
        use_cache: 是否使用内存缓存
        return_cache_info: 是否返回缓存信息

    Returns:
        BaseParakeet 或 tuple[BaseParakeet, bool]: 模型实例

    Raises:
        FileNotFoundError: 模型路径或文件不存在
    """

    def _load_model() -> BaseParakeet:
        """实际的模型加载逻辑"""
        # 获取模型路径
        path = _get_model_path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"模型路径不存在: {path}")

        config_path = path / "config.json"
        weight_path = path / "model.safetensors"

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        if not weight_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")

        logger.info(f"加载转录模型: {path}")

        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 构建模型
        model = from_config(config)

        # 加载权重
        model.load_weights(str(weight_path))

        # 转换数据类型
        curr_weights = dict(tree_flatten(model.parameters()))
        curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
        model.update(tree_unflatten(curr_weights))

        logger.info("模型加载完成")
        return model

    # 缓存键：使用实际的模型路径
    cache_key = model_path or str(_get_model_path(model_path))

    if use_cache:
        model, from_cache = load_cached_model(cache_key, dtype, _load_model)
        if return_cache_info:
            return model, from_cache
        return model
    else:
        model = _load_model()
        if return_cache_info:
            return model, False
        return model
