import json
import logging
from pathlib import Path
import time
import os

import mlx.core as mx
from dacite import from_dict
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten
import retry

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

# 设置日志记录器
logger = logging.getLogger(__name__)

def from_config(config: dict) -> BaseParakeet:
    """Loads model from config (randomized weight)"""
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
        raise ValueError("Model is not supported yet!")

    model.eval()  # prevents layernorm not computing correctly on inference!

    return model


@retry.retry(
    exceptions=(HfHubHTTPError, ConnectionError, TimeoutError, OSError),
    tries=3,
    delay=2,
    backoff=2,
    max_delay=10,
    logger=logger
)
def _download_with_retry(hf_id_or_path: str, filename: str) -> str:
    """
    带重试功能的文件下载函数
    
    Args:
        hf_id_or_path: Hugging Face 模型ID或路径
        filename: 要下载的文件名
        
    Returns:
        下载文件的本地路径
        
    Raises:
        RepositoryNotFoundError: 仓库不存在
        LocalEntryNotFoundError: 文件不存在
        Exception: 其他下载错误
    """
    logger.info(f"正在下载 {hf_id_or_path}/{filename}...")
    
    try:
        file_path = hf_hub_download(hf_id_or_path, filename)
        logger.info(f"下载成功: {filename}")
        return file_path
    except (RepositoryNotFoundError, LocalEntryNotFoundError) as e:
        # 这些错误不需要重试，直接抛出
        logger.error(f"文件不存在: {hf_id_or_path}/{filename}")
        raise e
    except Exception as e:
        logger.warning(f"下载失败，将重试: {str(e)}")
        raise e


def _find_cached_model(hf_id_or_path: str) -> tuple[str, str]:
    """
    查找已缓存的 Hugging Face 模型文件
    
    Args:
        hf_id_or_path: Hugging Face 模型ID
        
    Returns:
        tuple: (config_path, weight_path)
        
    Raises:
        FileNotFoundError: 找不到缓存的模型文件
    """
    # 尝试从环境变量或默认位置找到 HF 缓存目录
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or Path.home() / ".cache" / "huggingface"
    cache_dir = Path(cache_dir)
    
    # 构建模型缓存路径
    model_cache_name = hf_id_or_path.replace("/", "--")
    model_cache_dir = cache_dir / "hub" / f"models--{model_cache_name}"
    
    logger.info(f"正在查找缓存模型: {model_cache_dir}")
    
    if not model_cache_dir.exists():
        raise FileNotFoundError(f"模型缓存目录不存在: {model_cache_dir}")
    
    # 查找 snapshots 目录下的最新版本
    snapshots_dir = model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"模型快照目录不存在: {snapshots_dir}")
    
    # 获取最新的快照（按修改时间排序）
    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        raise FileNotFoundError(f"没有找到模型快照: {snapshots_dir}")
    
    latest_snapshot = max(snapshot_dirs, key=lambda d: d.stat().st_mtime)
    logger.info(f"找到最新快照: {latest_snapshot}")
    
    # 检查配置文件和权重文件
    config_path = latest_snapshot / "config.json"
    weight_path = latest_snapshot / "model.safetensors"
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    if not weight_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    
    logger.info(f"找到缓存的配置文件: {config_path}")
    logger.info(f"找到缓存的权重文件: {weight_path}")
    
    return str(config_path), str(weight_path)


@retry.retry(
    exceptions=(FileNotFoundError, PermissionError, OSError, json.JSONDecodeError),
    tries=3,
    delay=1,
    backoff=1.5,
    max_delay=5,
    logger=logger
)
def _load_model_files(config_path: str, weight_path: str) -> tuple[dict, str]:
    """
    带重试功能的模型文件加载函数
    
    Args:
        config_path: 配置文件路径
        weight_path: 权重文件路径
        
    Returns:
        tuple: (config_dict, weight_path)
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 解析错误
        Exception: 其他加载错误
    """
    logger.info(f"正在加载配置文件: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 检查权重文件是否存在
        if not Path(weight_path).exists():
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")
            
        logger.info("模型文件加载成功")
        return config, weight_path
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败，将重试: {str(e)}")
        raise e
    except Exception as e:
        logger.warning(f"文件加载失败，将重试: {str(e)}")
        raise e


def from_pretrained(
    hf_id_or_path: str, *, dtype: mx.Dtype = mx.bfloat16
) -> BaseParakeet:
    """
    从 Hugging Face 或本地目录加载模型，优先使用本地缓存
    
    加载策略（按优先级）：
    1. 优先查找本地缓存的模型文件
    2. 尝试从指定的本地路径加载  
    3. 最后才从 Hugging Face Hub 下载
    
    Args:
        hf_id_or_path: Hugging Face 模型ID或本地路径
        dtype: 模型数据类型
        
    Returns:
        加载的 Parakeet 模型
        
    Raises:
        ValueError: 不支持的模型类型
        FileNotFoundError: 配置文件或模型权重文件不存在
        Exception: 其他加载错误
    """
    logger.info(f"开始加载模型: {hf_id_or_path}")
    
    config = None
    weight = None
    
    # 策略1: 优先查找本地缓存的模型文件（最快，无网络请求）
    try:
        logger.info("策略1: 查找本地缓存的模型文件...")
        config_path, weight_path = _find_cached_model(hf_id_or_path)
        config, weight = _load_model_files(config_path, weight_path)
        logger.info("✅ 成功从本地缓存加载模型文件（无需下载）")
        
    except Exception as e:
        logger.info(f"本地缓存不可用: {str(e)}")
    
    # 策略2: 尝试从指定的本地路径加载
    if config is None:
        try:
            logger.info("策略2: 尝试从指定的本地路径加载...")
            local_path = Path(hf_id_or_path)
            config_path = str(local_path / "config.json")
            weight_path = str(local_path / "model.safetensors")
            
            config, weight = _load_model_files(config_path, weight_path)
            logger.info("✅ 成功从指定本地路径加载模型文件")
            
        except Exception as e:
            logger.info(f"指定本地路径不可用: {str(e)}")
    
    # 策略3: 最后才从 Hugging Face Hub 下载（需要网络连接）
    if config is None:
        try:
            logger.info("策略3: 从 Hugging Face Hub 下载模型文件...")
            logger.info("⚠️  本地未找到模型文件，开始在线下载（可能需要较长时间）...")
            config_path = _download_with_retry(hf_id_or_path, "config.json")
            weight_path = _download_with_retry(hf_id_or_path, "model.safetensors")
            
            config, weight = _load_model_files(config_path, weight_path)
            logger.info("✅ 成功从 Hugging Face Hub 下载并加载模型文件")
            
        except (RepositoryNotFoundError, LocalEntryNotFoundError):
            logger.error("Hugging Face Hub 中未找到指定模型")
            
        except Exception as e:
            logger.error(f"从 Hugging Face Hub 下载失败: {str(e)}")
    
    # 如果所有策略都失败了
    if config is None:
        error_msg = f"无法加载模型 {hf_id_or_path}。已尝试所有加载策略：\n" \
                   f"1. ❌ 本地缓存加载失败\n" \
                   f"2. ❌ 指定本地路径加载失败\n" \
                   f"3. ❌ 在线下载失败\n" \
                   f"请检查网络连接、模型ID是否正确，或确保模型文件存在。"
        logger.error(error_msg)
        raise Exception(error_msg)

    # 构建和加载模型
    try:
        logger.info("正在从配置构建模型...")
        model = from_config(config)
        
        logger.info("正在加载模型权重...")
        model.load_weights(weight)

        # cast dtype
        logger.info(f"正在转换模型数据类型为 {dtype}...")
        curr_weights = dict(tree_flatten(model.parameters()))
        curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
        model.update(tree_unflatten(curr_weights))
        
        logger.info("🎉 模型加载完成！")
        return model
        
    except Exception as e:
        error_msg = f"模型构建或权重加载失败: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
