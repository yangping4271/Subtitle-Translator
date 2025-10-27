import json
import logging
from pathlib import Path
import time
import os
import requests
import urllib.parse
import subprocess
import shutil
import pickle
import hashlib
from typing import Optional, List, Dict, Any, Union

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
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    FileSizeColumn,
)
from rich.text import Text

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
from .model_cache import load_cached_model, model_context
from ..logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)
console = Console()


# 存储层优化 - 预编译模型缓存
class ModelStorageOptimizer:
    """模型存储层优化器 - 通过缓存优化后的模型状态加速加载"""
    
    def __init__(self):
        # 获取缓存目录
        self.cache_root = self._get_cache_dir()
        self.optimized_cache_dir = self.cache_root / "optimized_models"
        self.optimized_cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_dir(self) -> Path:
        """获取缓存目录"""
        cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or Path.home() / ".cache" / "huggingface"
        return Path(cache_dir)
    
    def _get_cache_key(self, model_id: str, dtype: mx.Dtype) -> str:
        """生成缓存键"""
        # 使用模型ID和数据类型生成唯一键
        content = f"{model_id}_{dtype}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_optimized_paths(self, model_id: str, dtype: mx.Dtype) -> Dict[str, Path]:
        """获取优化缓存文件路径"""
        cache_key = self._get_cache_key(model_id, dtype)
        cache_dir = self.optimized_cache_dir / cache_key
        
        return {
            "cache_dir": cache_dir,
            "config_file": cache_dir / "config.json",
            "weights_file": cache_dir / "optimized_weights.safetensors", 
            "metadata_file": cache_dir / "metadata.json",
            "model_state_file": cache_dir / "model_state.pkl"
        }
    
    def has_optimized_cache(self, model_id: str, dtype: mx.Dtype) -> bool:
        """检查是否存在优化缓存"""
        paths = self._get_optimized_paths(model_id, dtype)
        return (
            paths["config_file"].exists() 
            and paths["weights_file"].exists()
            and paths["metadata_file"].exists()
        )
    
    def save_optimized_model(self, model_id: str, dtype: mx.Dtype, 
                           model: BaseParakeet, config: Dict[str, Any],
                           original_weight_path: str) -> None:
        """保存优化后的模型到存储"""
        try:
            paths = self._get_optimized_paths(model_id, dtype)
            paths["cache_dir"].mkdir(parents=True, exist_ok=True)
            
            # 1. 保存配置文件
            with open(paths["config_file"], 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # 2. 保存优化后的权重（已转换数据类型）
            import shutil
            shutil.copy2(original_weight_path, paths["weights_file"])
            
            # 3. 保存元数据
            metadata = {
                "model_id": model_id,
                "dtype": str(dtype),
                "cache_time": time.time(),
                "original_weight_path": original_weight_path,
                "version": "1.0"
            }
            with open(paths["metadata_file"], 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"已保存优化缓存: {model_id} ({dtype})")
            
        except Exception as e:
            logger.warning(f"保存优化缓存失败: {e}")
    
    def load_optimized_model(self, model_id: str, dtype: mx.Dtype) -> Optional[BaseParakeet]:
        """从存储加载优化的模型"""
        try:
            if not self.has_optimized_cache(model_id, dtype):
                return None
            
            paths = self._get_optimized_paths(model_id, dtype)
            
            # 加载配置
            with open(paths["config_file"], 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 构建模型
            model = from_config(config)
            
            # 加载权重
            model.load_weights(str(paths["weights_file"]))
            
            # 转换数据类型（可能已经是正确类型，但保险起见）
            curr_weights = dict(tree_flatten(model.parameters()))
            curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
            model.update(tree_unflatten(curr_weights))
            
            logger.info(f"从优化缓存加载模型成功: {model_id} ({dtype})")
            return model
            
        except Exception as e:
            logger.warning(f"从优化缓存加载模型失败: {e}")
            # 清理可能损坏的缓存
            self.clear_optimized_cache(model_id, dtype)
            return None
    
    def clear_optimized_cache(self, model_id: str, dtype: mx.Dtype) -> None:
        """清理特定模型的优化缓存"""
        try:
            paths = self._get_optimized_paths(model_id, dtype)
            if paths["cache_dir"].exists():
                shutil.rmtree(paths["cache_dir"])
                logger.info(f"已清理优化缓存: {model_id} ({dtype})")
        except Exception as e:
            logger.warning(f"清理优化缓存失败: {e}")
    
    def clear_all_optimized_cache(self) -> None:
        """清理所有优化缓存"""
        try:
            if self.optimized_cache_dir.exists():
                shutil.rmtree(self.optimized_cache_dir)
                self.optimized_cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("已清理所有优化缓存")
        except Exception as e:
            logger.warning(f"清理所有优化缓存失败: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取优化缓存统计信息"""
        try:
            if not self.optimized_cache_dir.exists():
                return {"cached_models": 0, "total_size": 0}
            
            cached_models = 0
            total_size = 0
            
            for cache_dir in self.optimized_cache_dir.iterdir():
                if cache_dir.is_dir():
                    metadata_file = cache_dir / "metadata.json"
                    if metadata_file.exists():
                        cached_models += 1
                        # 计算目录大小
                        for file in cache_dir.rglob('*'):
                            if file.is_file():
                                total_size += file.stat().st_size
            
            return {
                "cached_models": cached_models,
                "total_size": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "cache_dir": str(self.optimized_cache_dir)
            }
        except Exception as e:
            logger.warning(f"获取缓存统计失败: {e}")
            return {"cached_models": 0, "total_size": 0}


# 全局存储优化器实例
_storage_optimizer = ModelStorageOptimizer()


def _is_huggingface_cli_available() -> bool:
    """检查是否安装了 huggingface-cli"""
    try:
        result = subprocess.run(
            ["huggingface-cli", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False

def _check_network_connectivity() -> bool:
    """检查网络连接状态"""
    try:
        test_url = "https://huggingface.co"
        response = requests.get(test_url, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def _get_file_size(hf_id_or_path: str, filename: str) -> Optional[int]:
    """获取远程文件大小"""
    try:
        import huggingface_hub
        api = huggingface_hub.HfApi()
        repo_info = api.repo_info(hf_id_or_path)
        for sibling in repo_info.siblings:
            if sibling.rfilename == filename:
                return sibling.size
    except Exception:
        pass
    return None


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


def _download_with_huggingface_cli(hf_id_or_path: str, filename: str, show_progress: bool = True) -> Optional[str]:
    """
    使用 huggingface-cli 下载文件

    Args:
        hf_id_or_path: Hugging Face 模型ID
        filename: 要下载的文件名
        show_progress: 是否显示进度

    Returns:
        下载文件的本地路径，失败时返回 None
    """
    try:
        # 构建命令
        cmd = [
            "huggingface-cli",
            "download",
            hf_id_or_path,
            filename,
            "--quiet" if not show_progress else ""
        ]
        cmd = [arg for arg in cmd if arg]  # 过滤空字符串

        if show_progress:
            console.print(f"🚀 [bold blue]使用 huggingface-cli 下载:[/bold blue] {filename}")

        # 执行下载
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )

        if result.returncode == 0:
            # 解析输出获取文件路径
            output_lines = result.stdout.strip().split('\n')
            if output_lines and output_lines[-1]:
                file_path = output_lines[-1].strip()
                if Path(file_path).exists():
                    if show_progress:
                        console.print(f"✅ [bold green]huggingface-cli 下载成功:[/bold green] {filename}")
                    return file_path

        if show_progress:
            console.print(f"❌ [yellow]huggingface-cli 下载失败:[/yellow] {result.stderr or 'Unknown error'}")

        return None

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
        if show_progress:
            console.print(f"❌ [yellow]huggingface-cli 执行失败:[/yellow] {str(e)}")
        return None

def _download_with_hf_hub(hf_id_or_path: str, filename: str, show_progress: bool = True) -> Optional[str]:
    """
    使用 hf_hub_download 下载文件

    Args:
        hf_id_or_path: Hugging Face 模型ID
        filename: 要下载的文件名
        show_progress: 是否显示进度

    Returns:
        下载文件的本地路径，失败时返回 None
    """
    try:
        if show_progress:
            console.print(f"📦 [bold blue]使用 hf_hub_download 下载:[/bold blue] {filename}")

        # 使用默认 API 下载
        file_path = hf_hub_download(hf_id_or_path, filename)

        if show_progress:
            console.print(f"✅ [bold green]hf_hub_download 下载成功:[/bold green] {filename}")

        return file_path

    except Exception as e:
        if show_progress:
            console.print(f"❌ [yellow]hf_hub_download 下载失败:[/yellow] {str(e)}")
        return None

def _download_with_retry(hf_id_or_path: str, filename: str, show_progress: bool = True) -> str:
    """
    智能下载函数：自动选择最佳下载方式

    下载策略（按优先级）：
    1. 使用 huggingface-cli
    2. 使用 hf_hub_download

    Args:
        hf_id_or_path: Hugging Face 模型ID或路径
        filename: 要下载的文件名
        show_progress: 是否显示下载进度

    Returns:
        下载文件的本地路径

    Raises:
        RepositoryNotFoundError: 仓库不存在
        LocalEntryNotFoundError: 文件不存在
        Exception: 其他下载错误
    """

    if show_progress:
        console.print(f"\n🔄 [bold cyan]开始下载:[/bold cyan] [bold]{filename}[/bold]")
        console.print("📋 [dim]下载策略: huggingface-cli → hf_hub_download[/dim]\n")

    # 检查基本网络连接
    if not _check_network_connectivity():
        raise ConnectionError("无法连接到 Hugging Face，请检查网络连接")

    cli_available = _is_huggingface_cli_available()
    download_attempts = []

    # 策略1: 使用 huggingface-cli
    if cli_available:
        if show_progress:
            console.print(f"🚀 [bold blue]策略1: huggingface-cli[/bold blue]")

        try:
            result = _download_with_huggingface_cli(hf_id_or_path, filename, show_progress)
            if result:
                if show_progress:
                    console.print("✅ [bold green]策略1 成功![/bold green]")
                return result
        except Exception as e:
            download_attempts.append(f"策略1 (huggingface-cli): {str(e)}")

    # 策略2: 使用 hf_hub_download
    if show_progress:
        console.print(f"📦 [bold blue]策略2: hf_hub_download[/bold blue]")

    try:
        result = _download_with_hf_hub(hf_id_or_path, filename, show_progress)
        if result:
            if show_progress:
                console.print("✅ [bold green]策略2 成功![/bold green]")
            return result
    except Exception as e:
        download_attempts.append(f"策略2 (hf_hub_download): {str(e)}")

    # 所有策略都失败了
    if show_progress:
        console.print(f"\n❌ [bold red]所有 {len(download_attempts)} 种下载策略均已尝试完毕[/bold red]")

    error_summary = "\n".join([f"   • {attempt}" for attempt in download_attempts])
    error_msg = f"""❌ [bold red]所有下载策略均失败[/bold red]

📋 [bold]尝试的下载方式:[/bold]
{error_summary}

💡 [bold yellow]解决建议:[/bold yellow]
   • 检查网络连接是否稳定
   • 确认模型ID是否正确: [cyan]{hf_id_or_path}[/cyan]
   • 尝试手动访问: [link]https://huggingface.co/{hf_id_or_path}[/link]"""

    if show_progress:
        console.print(error_msg)

    # 尝试判断具体的错误类型
    for attempt in download_attempts:
        if "404" in attempt or "Repository not found" in attempt:
            raise RepositoryNotFoundError(f"模型仓库不存在: {hf_id_or_path}")
        elif "File not found" in attempt or f"{filename}" in attempt:
            raise LocalEntryNotFoundError(f"文件不存在: {hf_id_or_path}/{filename}")

    raise Exception(f"无法下载文件 {filename} 从 {hf_id_or_path}")


def _find_cached_model(hf_id_or_path: str) -> tuple[str, str]:
    """
    查找已缓存的 Hugging Face 模型文件
    优先检查存储优化缓存，然后检查标准HF缓存
    
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
    
    # 优先检查存储优化缓存
    optimized_cache_dir = cache_dir / "optimized_models"
    if optimized_cache_dir.exists():
        import mlx.core as mx
        from . import utils
        try:
            storage_optimizer = utils._storage_optimizer
            cache_key = storage_optimizer._get_cache_key(hf_id_or_path, mx.bfloat16)
            optimized_model_dir = optimized_cache_dir / cache_key
            
            if optimized_model_dir.exists():
                config_path = optimized_model_dir / "config.json"
                weight_path = optimized_model_dir / "optimized_weights.safetensors"
                
                if config_path.exists() and weight_path.exists():
                    logger.info(f"从存储优化缓存找到模型: {optimized_model_dir}")
                    return str(config_path), str(weight_path)
        except Exception as e:
            logger.info(f"存储优化缓存检查失败: {e}")
    
    # 构建标准模型缓存路径
    model_cache_name = hf_id_or_path.replace("/", "--")
    model_cache_dir = cache_dir / "hub" / f"models--{model_cache_name}"
    
    logger.info(f"正在查找标准缓存模型: {model_cache_dir}")
    
    if not model_cache_dir.exists():
        raise FileNotFoundError(f"模型缓存目录不存在（标准缓存: {model_cache_dir}，优化缓存: {optimized_cache_dir}）")
    
    # 查找 snapshots 目录下的最新版本
    snapshots_dir = model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"模型快照目录不存在")
    
    # 获取最新的快照（按修改时间排序）
    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        raise FileNotFoundError(f"没有找到模型快照")
    
    latest_snapshot = max(snapshot_dirs, key=lambda d: d.stat().st_mtime)
    logger.info(f"找到最新快照: {latest_snapshot}")
    
    # 检查配置文件和权重文件
    config_path = latest_snapshot / "config.json"
    weight_path = latest_snapshot / "model.safetensors"
    
    if not config_path.exists():
        raise FileNotFoundError(f"缓存的配置文件不存在")
    if not weight_path.exists():
        raise FileNotFoundError(f"缓存的权重文件不存在或未完整下载")
    
    logger.info(f"找到缓存的配置文件: {config_path}")
    logger.info(f"找到缓存的权重文件: {weight_path}")
    
    return str(config_path), str(weight_path)


def _load_model_files(config_path: str, weight_path: str, silent: bool = False) -> tuple[dict, str]:
    """
    模型文件加载函数，具有用户友好的错误处理
    
    Args:
        config_path: 配置文件路径
        weight_path: 权重文件路径
        silent: 是否禁用详细的错误输出
        
    Returns:
        tuple: (config_dict, weight_path)
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 解析错误
        Exception: 其他加载错误
    """
    
    # 检查配置文件
    if not Path(config_path).exists():
        if not silent:
            logger.info("配置文件不存在，将尝试在线下载")
        raise FileNotFoundError(f"配置文件不存在: {Path(config_path).name}")
    
    # 检查权重文件
    if not Path(weight_path).exists():
        if not silent:
            logger.info("模型权重文件不存在，将尝试在线下载")
        raise FileNotFoundError(f"权重文件不存在: {Path(weight_path).name}")
    
    try:
        logger.info(f"正在加载配置文件: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        logger.info("模型文件加载成功")
        return config, weight_path
        
    except json.JSONDecodeError as e:
        if not silent:
            logger.warning("配置文件格式错误，可能已损坏")
        raise json.JSONDecodeError("配置文件格式错误", config_path, 0) from e
    except Exception as e:
        if not silent:
            logger.warning("模型文件访问失败")
        raise e


def from_pretrained(
    hf_id_or_path: str, *, dtype: mx.Dtype = mx.bfloat16, show_progress: bool = True, 
    use_cache: bool = True, return_cache_info: bool = False
) -> Union[BaseParakeet, tuple[BaseParakeet, bool]]:
    """
    从 Hugging Face 或本地目录加载模型，支持内存和存储层缓存优化
    
    加载策略（按优先级）：
    1. 内存缓存（最快，毫秒级）
    2. 存储层优化缓存（快，秒级）
    3. 本地缓存的原始模型文件（中等）
    4. 指定的本地路径加载（中等）
    5. 从 Hugging Face Hub 下载（最慢）
    
    Args:
        hf_id_or_path: Hugging Face 模型ID或本地路径
        dtype: 模型数据类型
        show_progress: 是否显示详细的加载进度
        use_cache: 是否使用缓存优化
        return_cache_info: 是否返回缓存信息 (model, from_cache)
        
    Returns:
        BaseParakeet 或 tuple[BaseParakeet, bool]: 模型实例，如果 return_cache_info=True 则返回 (模型, 是否从缓存加载)
        
    Raises:
        ValueError: 不支持的模型类型
        FileNotFoundError: 配置文件或模型权重文件不存在
        Exception: 其他加载错误
    """
    
    def _original_loader() -> BaseParakeet:
        """原始的模型加载逻辑，作为fallback"""
        return _load_model_original(hf_id_or_path, dtype, show_progress, skip_optimized_cache=True)
    
    # 使用缓存优化的加载
    if use_cache:
        model, from_cache = load_cached_model(hf_id_or_path, dtype, _original_loader, show_progress)
        if return_cache_info:
            return model, from_cache
        return model
    else:
        # 不使用缓存，直接加载
        model = _original_loader()
        if return_cache_info:
            return model, False
        return model


def _load_model_original(
    hf_id_or_path: str, dtype: mx.Dtype = mx.bfloat16, show_progress: bool = True, skip_optimized_cache: bool = False
) -> BaseParakeet:
    """
    原始模型加载逻辑（重构后的内部函数）
    
    加载策略（按优先级）：
    1. 存储层优化缓存（预编译模型）
    2. 本地缓存的原始模型文件
    3. 指定的本地路径加载
    4. 从 Hugging Face Hub 下载
    """
    if show_progress:
        console.print(f"\n🤖 [bold cyan]开始加载模型:[/bold cyan] [bold]{hf_id_or_path}[/bold]")
        console.print("📋 [dim]加载策略: 存储优化缓存 → 本地缓存 → 本地路径 → 在线下载[/dim]\n")
    
    config = None
    weight = None
    loading_method = None
    
    # 策略1: 尝试从存储层优化缓存加载（最快的文件加载）
    if not skip_optimized_cache:
        try:
            if show_progress:
                with console.status("[bold blue]🚀 策略1: 查找存储层优化缓存...[/bold blue]"):
                    time.sleep(0.2)  # 给用户一点时间看到状态
                    optimized_model = _storage_optimizer.load_optimized_model(hf_id_or_path, dtype)
            else:
                optimized_model = _storage_optimizer.load_optimized_model(hf_id_or_path, dtype)
            
            if optimized_model is not None:
                loading_method = "存储优化缓存"
                if show_progress:
                    console.print("✅ [bold green]从存储优化缓存加载成功![/bold green] (极速加载)")
                    console.print(f"\n🎉 [bold green]模型加载完成![/bold green] (加载方式: [bold cyan]{loading_method}[/bold cyan])")
                    console.print("━" * 60)
                return optimized_model
            else:
                if show_progress:
                    console.print("🔍 [dim]存储优化缓存不可用，将尝试其他方式[/dim]")
        except Exception as e:
            if show_progress:
                console.print(f"🔍 [dim]存储优化缓存查找失败: {str(e)}[/dim]")
            logger.info(f"存储优化缓存查找失败: {str(e)}")
    else:
        # 存储优化缓存已在上层检查过，直接跳过
        if show_progress:
            console.print("🔍 [dim]跳过存储优化缓存检查（已在缓存层处理）[/dim]")
    
    config = None
    weight = None
    
    # 策略2: 查找本地缓存的原始模型文件（无网络请求）
    try:
        if show_progress:
            with console.status("[bold blue]🔍 策略2: 查找本地缓存的模型文件...[/bold blue]"):
                time.sleep(0.3)  # 给用户一点时间看到状态
                config_path, weight_path = _find_cached_model(hf_id_or_path)
                config, weight = _load_model_files(config_path, weight_path, silent=not show_progress)
        else:
            config_path, weight_path = _find_cached_model(hf_id_or_path)
            config, weight = _load_model_files(config_path, weight_path, silent=True)
        
        loading_method = "本地缓存"
        if show_progress:
            console.print("✅ [bold green]成功从本地缓存加载模型文件[/bold green] (无需下载)")
        
    except Exception as e:
        if show_progress:
            console.print(f"🔍 [dim]本地缓存不可用，将尝试其他方式[/dim]")
        logger.info(f"本地缓存查找失败: {str(e)}")
    
    # 策略3: 尝试从指定的本地路径加载
    if config is None:
        try:
            if show_progress:
                with console.status("[bold blue]🔍 策略3: 尝试从指定的本地路径加载...[/bold blue]"):
                    local_path = Path(hf_id_or_path)
                    config_path = str(local_path / "config.json")
                    weight_path = str(local_path / "model.safetensors")
                    config, weight = _load_model_files(config_path, weight_path, silent=not show_progress)
            else:
                local_path = Path(hf_id_or_path)
                config_path = str(local_path / "config.json")
                weight_path = str(local_path / "model.safetensors")
                config, weight = _load_model_files(config_path, weight_path, silent=True)
            
            loading_method = "本地路径"
            if show_progress:
                console.print("✅ [bold green]成功从指定本地路径加载模型文件[/bold green]")
            
        except Exception as e:
            if show_progress:
                console.print(f"🔍 [dim]指定本地路径不可用，将尝试在线下载[/dim]")
            logger.info(f"本地路径加载失败: {str(e)}")
    
    # 策略4: 最后才从 Hugging Face Hub 下载（需要网络连接）
    if config is None:
        try:
            if show_progress:
                console.print("\n⚠️  [bold yellow]本地未找到模型文件，开始在线下载[/bold yellow]")
                console.print(f"📦 [bold]模型信息:[/bold] [cyan]{hf_id_or_path}[/cyan]")
                console.print("📏 [bold]预计大小:[/bold] ~1.2GB")
                console.print("⏱️  [bold]预计时间:[/bold] 3-10分钟 (取决于网络速度)")
                console.print("💡 [dim]提示: 首次下载较大，后续使用将直接从缓存加载[/dim]")
                console.print("🔄 [dim]下载中断可以重新运行命令继续下载[/dim]")
                
                # 检查网络连接
                with console.status("[bold blue]🌐 检查网络连接...[/bold blue]"):
                    if not _check_network_connectivity():
                        raise ConnectionError("无法连接到 Hugging Face Hub")
                
                console.print("✅ [green]网络连接正常[/green]")
                
            config_path = _download_with_retry(hf_id_or_path, "config.json", show_progress)
            weight_path = _download_with_retry(hf_id_or_path, "model.safetensors", show_progress)
            
            config, weight = _load_model_files(config_path, weight_path, silent=not show_progress)
            loading_method = "在线下载"
            if show_progress:
                console.print("\n✅ [bold green]模型下载并加载成功！[/bold green]")
                console.print("🎉 [dim]模型已缓存到本地，后续使用将更快[/dim]")
            
        except (RepositoryNotFoundError, LocalEntryNotFoundError):
            error_msg = f"❌ Hugging Face Hub 中未找到指定模型: [bold red]{hf_id_or_path}[/bold red]"
            if show_progress:
                console.print(error_msg)
            logger.error(error_msg)
            
        except ConnectionError as e:
            error_msg = f"❌ 网络连接失败: {str(e)}"
            if show_progress:
                console.print(f"[bold red]{error_msg}[/bold red]")
                console.print("💡 [dim]建议：[/dim]")
                console.print("   • 检查网络连接")
                console.print("   • 确认防火墙设置允许访问 huggingface.co")
            logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"❌ 从 Hugging Face Hub 下载失败: {str(e)}"
            if show_progress:
                console.print(f"[bold red]{error_msg}[/bold red]")
            logger.error(error_msg)
    
    # 如果所有策略都失败了
    if config is None:
        error_msg = f"""❌ [bold red]无法加载模型 {hf_id_or_path}[/bold red]

📋 [bold]已尝试的加载策略:[/bold]
   1. ❌ 存储优化缓存加载失败
   2. ❌ 本地缓存加载失败
   3. ❌ 指定本地路径加载失败  
   4. ❌ 在线下载失败

💡 [bold yellow]解决建议:[/bold yellow]
   • 检查网络连接是否正常
   • 确认模型ID是否正确: [cyan]{hf_id_or_path}[/cyan]
   • 如果是本地路径，确保模型文件存在
   • 尝试手动访问: [link]https://huggingface.co/{hf_id_or_path}[/link]"""
        
        if show_progress:
            console.print(error_msg)
        
        logger.error(f"所有模型加载策略均失败: {hf_id_or_path}")
        raise Exception(f"无法加载模型 {hf_id_or_path}")

    # 构建和加载模型
    try:
        if show_progress:
            console.print(f"\n🔧 [bold blue]正在构建模型...[/bold blue]")
            with console.status("[bold blue]解析配置文件...[/bold blue]"):
                model = from_config(config)
        else:
            model = from_config(config)
        
        if show_progress:
            console.print("✅ [green]模型构建成功[/green]")
            
            with console.status("[bold blue]🔗 正在加载模型权重...[/bold blue]"):
                model.load_weights(weight)
            console.print("✅ [green]模型权重加载成功[/green]")
        else:
            model.load_weights(weight)

        # 转换数据类型
        if show_progress:
            with console.status(f"[bold blue]🔄 正在转换模型数据类型为 {dtype}...[/bold blue]"):
                curr_weights = dict(tree_flatten(model.parameters()))
                curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
                model.update(tree_unflatten(curr_weights))
            console.print(f"✅ [green]数据类型转换完成[/green] ({dtype})")
        else:
            curr_weights = dict(tree_flatten(model.parameters()))
            curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
            model.update(tree_unflatten(curr_weights))
        
        # 如果是从原始文件加载且不是存储优化缓存，尝试保存优化缓存（异步，不影响主流程）
        if loading_method in ["本地缓存", "本地路径", "在线下载"]:
            try:
                if show_progress:
                    with console.status("[bold blue]💾 保存存储优化缓存...[/bold blue]"):
                        _storage_optimizer.save_optimized_model(hf_id_or_path, dtype, model, config, weight)
                    console.print("✅ [green]存储优化缓存已保存[/green] (下次加载将更快)")
                else:
                    _storage_optimizer.save_optimized_model(hf_id_or_path, dtype, model, config, weight)
                    logger.info("存储优化缓存已保存")
            except Exception as e:
                # 保存缓存失败不影响主流程
                logger.info(f"保存存储优化缓存失败: {e}")
                if show_progress:
                    console.print("⚠️  [yellow]存储优化缓存保存失败，不影响模型使用[/yellow]")
        
        if show_progress:
            console.print(f"\n🎉 [bold green]模型加载完成![/bold green] (加载方式: [bold cyan]{loading_method}[/bold cyan])")
            console.print("━" * 60)
        
        return model
        
    except Exception as e:
        error_msg = f"❌ 模型构建或权重加载失败: {str(e)}"
        if show_progress:
            console.print(f"[bold red]{error_msg}[/bold red]")
        logger.error(error_msg)
        raise Exception(error_msg) from e
