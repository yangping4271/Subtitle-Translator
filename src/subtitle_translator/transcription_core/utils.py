import json
import logging
from pathlib import Path
import time
import os
import requests
import urllib.parse
from typing import Optional

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

# 设置日志记录器
logger = logging.getLogger(__name__)
console = Console()

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


def _check_network_connectivity() -> bool:
    """检查网络连接状态"""
    try:
        response = requests.get("https://huggingface.co", timeout=5)
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


@retry.retry(
    exceptions=(HfHubHTTPError, ConnectionError, TimeoutError, OSError),
    tries=3,
    delay=2,
    backoff=2,
    max_delay=10,
    logger=logger
)
def _download_with_retry(hf_id_or_path: str, filename: str, show_progress: bool = True) -> str:
    """
    带重试功能和进度显示的文件下载函数
    
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
    # 检查网络连接
    if not _check_network_connectivity():
        raise ConnectionError("无法连接到 Hugging Face Hub，请检查网络连接")
    
    # 获取文件大小用于进度显示
    file_size = _get_file_size(hf_id_or_path, filename) if show_progress else None
    
    if show_progress and file_size:
        # 显示文件大小信息
        size_mb = file_size / (1024 * 1024)
        console.print(f"📦 开始下载 [bold blue]{filename}[/bold blue] (大小: {size_mb:.1f} MB)")
        
        # 使用 Rich 进度条显示下载进度
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.1f}%",
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:
            task = progress.add_task(f"下载 {filename}", total=file_size)
            
            def progress_callback(chunk_size: int):
                progress.update(task, advance=chunk_size)
            
            try:
                # 这里我们使用原有的 hf_hub_download，但在实际实现中
                # 可能需要自定义下载逻辑来支持进度回调
                file_path = hf_hub_download(hf_id_or_path, filename)
                progress.update(task, completed=file_size)
                console.print(f"✅ 下载完成: [bold green]{filename}[/bold green]")
                return file_path
            except (RepositoryNotFoundError, LocalEntryNotFoundError) as e:
                progress.stop()
                logger.error(f"文件不存在: {hf_id_or_path}/{filename}")
                raise e
            except Exception as e:
                progress.stop()
                logger.warning(f"下载失败，将重试: {str(e)}")
                raise e
    else:
        # 简单模式，不显示详细进度
        console.print(f"📦 正在下载 [bold blue]{filename}[/bold blue]...")
        
        try:
            file_path = hf_hub_download(hf_id_or_path, filename)
            console.print(f"✅ 下载完成: [bold green]{filename}[/bold green]")
            return file_path
        except (RepositoryNotFoundError, LocalEntryNotFoundError) as e:
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
    hf_id_or_path: str, *, dtype: mx.Dtype = mx.bfloat16, show_progress: bool = True
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
        show_progress: 是否显示详细的加载进度
        
    Returns:
        加载的 Parakeet 模型
        
    Raises:
        ValueError: 不支持的模型类型
        FileNotFoundError: 配置文件或模型权重文件不存在
        Exception: 其他加载错误
    """
    if show_progress:
        console.print(f"\n🤖 [bold cyan]开始加载模型:[/bold cyan] [bold]{hf_id_or_path}[/bold]")
        console.print("📋 [dim]加载策略: 本地缓存 → 本地路径 → 在线下载[/dim]\n")
    
    config = None
    weight = None
    loading_method = None
    
    # 策略1: 优先查找本地缓存的模型文件（最快，无网络请求）
    try:
        if show_progress:
            with console.status("[bold blue]🔍 策略1: 查找本地缓存的模型文件...[/bold blue]"):
                time.sleep(0.5)  # 给用户一点时间看到状态
                config_path, weight_path = _find_cached_model(hf_id_or_path)
                config, weight = _load_model_files(config_path, weight_path)
        else:
            config_path, weight_path = _find_cached_model(hf_id_or_path)
            config, weight = _load_model_files(config_path, weight_path)
        
        loading_method = "本地缓存"
        if show_progress:
            console.print("✅ [bold green]成功从本地缓存加载模型文件[/bold green] (无需下载)")
        
    except Exception as e:
        if show_progress:
            console.print(f"❌ [yellow]本地缓存不可用:[/yellow] [dim]{str(e)}[/dim]")
    
    # 策略2: 尝试从指定的本地路径加载
    if config is None:
        try:
            if show_progress:
                with console.status("[bold blue]🔍 策略2: 尝试从指定的本地路径加载...[/bold blue]"):
                    local_path = Path(hf_id_or_path)
                    config_path = str(local_path / "config.json")
                    weight_path = str(local_path / "model.safetensors")
                    config, weight = _load_model_files(config_path, weight_path)
            else:
                local_path = Path(hf_id_or_path)
                config_path = str(local_path / "config.json")
                weight_path = str(local_path / "model.safetensors")
                config, weight = _load_model_files(config_path, weight_path)
            
            loading_method = "本地路径"
            if show_progress:
                console.print("✅ [bold green]成功从指定本地路径加载模型文件[/bold green]")
            
        except Exception as e:
            if show_progress:
                console.print(f"❌ [yellow]指定本地路径不可用:[/yellow] [dim]{str(e)}[/dim]")
    
    # 策略3: 最后才从 Hugging Face Hub 下载（需要网络连接）
    if config is None:
        try:
            if show_progress:
                console.print("\n⚠️  [bold yellow]本地未找到模型文件，开始在线下载[/bold yellow]")
                console.print("💡 [dim]提示: 首次下载可能需要较长时间，请耐心等待...[/dim]")
                
                # 检查网络连接
                with console.status("[bold blue]🌐 检查网络连接...[/bold blue]"):
                    if not _check_network_connectivity():
                        raise ConnectionError("无法连接到 Hugging Face Hub")
                
                console.print("✅ [green]网络连接正常[/green]")
                
            config_path = _download_with_retry(hf_id_or_path, "config.json", show_progress)
            weight_path = _download_with_retry(hf_id_or_path, "model.safetensors", show_progress)
            
            config, weight = _load_model_files(config_path, weight_path)
            loading_method = "在线下载"
            if show_progress:
                console.print("\n✅ [bold green]成功从 Hugging Face Hub 下载并加载模型文件[/bold green]")
            
        except (RepositoryNotFoundError, LocalEntryNotFoundError):
            error_msg = f"❌ Hugging Face Hub 中未找到指定模型: [bold red]{hf_id_or_path}[/bold red]"
            if show_progress:
                console.print(error_msg)
            logger.error(error_msg)
            
        except ConnectionError as e:
            error_msg = f"❌ 网络连接失败: {str(e)}"
            if show_progress:
                console.print(f"[bold red]{error_msg}[/bold red]")
                console.print("💡 [dim]请检查网络连接或稍后重试[/dim]")
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
   1. ❌ 本地缓存加载失败
   2. ❌ 指定本地路径加载失败  
   3. ❌ 在线下载失败

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

        # cast dtype
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
