"""
配置管理模块 - 配置验证和初始化
"""
import os
from pathlib import Path
from functools import wraps

import typer
import click
from rich import print

from .env_setup import setup_environment, get_app_config_dir, get_global_env_path
from .translation_core.utils.test_openai import test_openai


def handle_user_abort(exit_message="❌ 配置已取消"):
    """装饰器：统一处理用户中断操作"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (KeyboardInterrupt, typer.Abort, click.exceptions.Abort):
                print(f"\n{exit_message}")
                import sys
                sys.exit(0)
        return wrapper
    return decorator


@handle_user_abort()
def safe_prompt(message, **kwargs):
    """安全的 typer.prompt 包装函数，自动处理用户取消操作"""
    return typer.prompt(message, **kwargs)


@handle_user_abort("❌ 操作已取消")
def safe_prompt_operation(message, **kwargs):
    """安全的 typer.prompt 包装函数（用于操作类提示），自动处理用户取消操作"""
    return typer.prompt(message, **kwargs)


def validate_existing_config_and_return_result(env_path: Path = None):
    """验证现有配置中的所有模型，返回验证结果"""
    try:
        # 重新加载环境变量
        from .env_setup import _env_loaded
        
        if env_path and env_path.exists():
            # 临时加载指定的环境文件
            from dotenv import load_dotenv
            load_dotenv(env_path, override=True)
        else:
            setup_environment()
        
        # 测试API连接
        base_url = os.getenv('OPENAI_BASE_URL')
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not base_url or not api_key:
            print("❌ 缺少必需的 API 配置 (OPENAI_BASE_URL 或 OPENAI_API_KEY)")
            return False
        
        # 获取所有需要验证的模型
        split_model = os.getenv('SPLIT_MODEL')
        translation_model = os.getenv('TRANSLATION_MODEL')
        summary_model = os.getenv('SUMMARY_MODEL')
        llm_model = os.getenv('LLM_MODEL')
        
        # 收集所有不同的模型
        unique_models = {}
        if split_model:
            unique_models['断句模型'] = split_model
        if translation_model:
            unique_models['翻译模型'] = translation_model
        if summary_model:
            unique_models['总结模型'] = summary_model
        if llm_model:
            unique_models['默认模型'] = llm_model
            
        if not unique_models:
            print("⚠️  未找到任何模型配置")
            return False
            
        # 去重：只测试不同的模型
        tested_models = set()
        validation_results = []
        
        for model_type, model_name in unique_models.items():
            if model_name not in tested_models:
                print(f"🔌 测试 {model_name}...")
                success, message = test_openai(base_url, api_key, model_name)
                tested_models.add(model_name)
                
                validation_results.append({
                    'model': model_name,
                    'success': success,
                    'message': message,
                    'types': [model_type]
                })
            else:
                # 如果模型已经测试过，找到之前的结果并添加类型
                for result in validation_results:
                    if result['model'] == model_name:
                        result['types'].append(model_type)
                        break
        
        # 显示验证结果
        print("\n📊 [bold blue]验证结果:[/bold blue]")
        all_success = True
        
        for result in validation_results:
            model_types = '、'.join(result['types'])
            if result['success']:
                print(f"   ✅ {result['model']} ({model_types})")
                print(f"      响应: {result['message'][:60]}...")
            else:
                print(f"   ❌ {result['model']} ({model_types})")
                print(f"      错误: {result['message']}")
                all_success = False
        
        return all_success
            
    except Exception as e:
        print(f"⚠️  配置验证过程中出现错误: {e}")
        return False


def validate_existing_config(env_path: Path = None):
    """验证现有配置中的所有模型（仅显示结果，不返回）"""
    result = validate_existing_config_and_return_result(env_path)
    if result:
        print("\n🎉 [bold green]所有模型验证成功！[/bold green]")
    else:
        print("\n⚠️  [bold yellow]部分模型验证失败，请检查模型名称和网络连接[/bold yellow]")


def _check_model_availability(model_id: str, detailed: bool = True) -> tuple[bool, str]:
    """
    检查转录模型可用性的统一函数
    
    Args:
        model_id: 转录模型ID
        detailed: 是否返回详细信息（包含缓存信息、ImportError处理等）
        
    Returns:
        tuple: (是否可用, 状态描述)
    """
    try:
        from .transcription_core.utils import _find_cached_model
        
        try:
            config_path, weight_path = _find_cached_model(model_id)
            if detailed:
                # 获取缓存目录名作为状态信息
                cache_info = Path(config_path).parent.parent.name
                return True, f"已缓存 ({cache_info})"
            else:
                return True, "已缓存"
        except FileNotFoundError:
            return False, "未下载" if detailed else "未缓存"
            
    except ImportError:
        # 只在非详细模式下特别处理ImportError
        if not detailed:
            return False, "模块未可用"
        else:
            return False, "检测失败: 转录模块导入失败"
    except Exception as e:
        if detailed:
            # 详细模式下截断错误信息
            return False, f"检测失败: {str(e)[:50]}..."
        else:
            # 简化模式下显示完整错误信息
            return False, f"检查失败: {str(e)}"


def _check_transcription_model_availability(model_id: str = "mlx-community/parakeet-tdt-0.6b-v2") -> tuple[bool, str]:
    """
    检查转录模型可用性 (兼容性包装函数)
    
    Args:
        model_id: 转录模型ID
        
    Returns:
        tuple: (是否可用, 状态描述)
    """
    return _check_model_availability(model_id, detailed=True)


def _display_model_download_guide(model_id: str = "mlx-community/parakeet-tdt-0.6b-v2"):
    """显示转录模型手动下载指南"""
    print(f"\n📋 [bold blue]转录模型下载指南:[/bold blue]")
    print("如需使用转录功能，可通过以下方式下载模型：")
    print("")
    print("1. 🔧 [bold]自动下载 (推荐):[/bold]")
    print("   首次运行转录命令时自动下载")
    print("   [dim]translate -i audio.mp3[/dim]")
    print("")
    print("2. 🌐 [bold]在线预下载:[/bold]")
    print("   通过 huggingface-cli 工具预下载")
    print(f"   [dim]huggingface-cli download {model_id}[/dim]")
    print("")
    print("3. 🏗️  [bold]镜像站下载 (国内用户推荐):[/bold]")
    print("   配置镜像站后下载更快更稳定")
    print("   [dim]export HF_ENDPOINT=https://hf-mirror.com[/dim]")
    print(f"   [dim]huggingface-cli download {model_id}[/dim]")
    print("")
    print("4. 📁 [bold]本地路径加载:[/bold]")
    print("   支持使用本地目录中的模型文件")
    print("   要求目录包含: config.json 和 model.safetensors")
    print("   [dim]translate -i audio.mp3 --model /path/to/local/model[/dim]")
    print("   💡 [dim]适用于预先下载的模型或自定义模型目录[/dim]")
    print("")
    print("5. 🔄 [bold]中断恢复:[/bold]")
    print("   如果下载被中断，可以重新运行任意下载命令继续")
    print("   [dim]translate -i audio.mp3  # 自动继续下载[/dim]")
    print(f"   [dim]huggingface-cli download {model_id}  # 手动继续下载[/dim]")
    print("   [dim]translate init  # 通过配置向导继续下载[/dim]")
    print("")
    print("6. 🗂️  [bold]缓存位置:[/bold]")
    print("   下载的模型缓存在 ~/.cache/huggingface/")
    print("   如需重新完整下载，可删除对应缓存目录")
    print("")
    print("💡 [dim]模型大小约 1.2GB，首次下载需要一些时间[/dim]")
    print("🔄 [dim]支持断点续传，下载中断后可以继续下载[/dim]")


def _display_system_status_summary():
    """显示系统状态总结"""
    print("\n📊 [bold green]系统状态总结:[/bold green]")
    
    # API 配置状态
    print("   🔑 API 配置: ✅ 已配置")
    
    # 转录模型状态
    model_available, model_status = _check_transcription_model_availability()
    status_icon = "✅" if model_available else "⚠️ "
    print(f"   🤖 转录模型: {status_icon} {model_status}")
    
    # 功能可用性
    print("   📝 翻译功能: ✅ 可用")
    transcription_status = "✅ 可用" if model_available else "⚠️  首次使用时下载"
    print(f"   🎙️  转录功能: {transcription_status}")


def _handle_model_download_suggestion():
    """处理转录模型下载建议"""
    model_available, model_status = _check_transcription_model_availability()
    
    if not model_available:
        print(f"\n💡 [bold yellow]发现转录模型未下载[/bold yellow]")
        print("转录功能需要下载默认模型 (mlx-community/parakeet-tdt-0.6b-v2)")
        print("模型大小约 1.2GB，建议在网络良好时预下载")
        print("🔄 [dim]如果之前下载被中断，现在可以继续完成下载[/dim]")
        
        # 检查网络连接
        try:
            from .transcription_core.utils import _check_network_connectivity
            has_network = _check_network_connectivity()
            
            if has_network:
                print("\n🔧 [bold blue]下载选项:[/bold blue]")
                print("   1. 现在预下载 (推荐)")
                print("   2. 跳过预下载")
                print("   3. 查看所有下载方式")
                
                while True:
                    choice = safe_prompt_operation(
                        "请选择 (1-3)", 
                        default="1", 
                        show_default=False
                    )
                    
                    if choice == "1":
                        # 预下载
                        print("\n" + "="*50)
                        return _execute_predownload()
                    elif choice == "2":
                        # 跳过预下载
                        print("⏭️  [dim]跳过预下载，首次使用时会自动下载[/dim]")
                        return False
                    elif choice == "3":
                        # 显示完整下载指南
                        _display_model_download_guide()
                        
                        # 显示指南后再次询问是否要预下载
                        followup_choice = safe_prompt_operation(
                            "\n📥 现在是否要预下载默认模型? (y/N)", 
                            default="n", 
                            show_default=False
                        ).lower()
                        
                        if followup_choice in ['y', 'yes', '是', '确定']:
                            print("\n" + "="*50)
                            return _execute_predownload()
                        else:
                            print("⏭️  [dim]跳过预下载，可以按照上述指南手动下载[/dim]")
                            return False
                    else:
                        print("❌ 请输入有效选择 (1-3)")
                        continue
            else:
                print("\n❌ [bold red]网络连接不可用[/bold red]")
                print("💡 [dim]网络恢复后可重新运行下载命令继续[/dim]")
                _display_model_download_guide()
                return False
        except Exception:
            # 网络检测失败，提供手动下载指南
            print("💡 [dim]如果之前有下载中断，可重新运行命令继续[/dim]")
            _display_model_download_guide()
            return False
    
    return True


def init_config():
    """初始化全局配置 - 智能检测并处理当前目录和全局配置的各种组合情况"""
    print("[bold green]🚀 字幕翻译工具配置初始化[/bold green]")
    
    # 获取全局配置目录和文件路径
    app_dir = get_app_config_dir()
    global_env_path = get_global_env_path()
    current_env_path = Path(".env")
    
    # 确保全局配置目录存在
    app_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查配置文件存在情况
    current_exists = current_env_path.exists()
    global_exists = global_env_path.exists()
    
    print(f"\n📁 [bold blue]配置文件检测结果:[/bold blue]")
    print(f"   📄 当前目录配置: {'✅ 存在' if current_exists else '❌ 不存在'} ([cyan].env[/cyan])")
    print(f"   🌐 全局配置: {'✅ 存在' if global_exists else '❌ 不存在'} ([cyan]{global_env_path}[/cyan])")
    
    default_model = "mlx-community/parakeet-tdt-0.6b-v2"
    
    # 检查默认转录模型是否可用
    model_available, model_status = _check_model_availability(default_model, detailed=False)
    
    if model_available:
        print(f"   ✅ 默认转录模型: {model_status} ([cyan]{default_model}[/cyan])")
    else:
        print(f"   ❌ 默认转录模型: {model_status} ([cyan]{default_model}[/cyan])")
        print(f"   📝 [dim]首次使用转录功能时会自动下载 (~1.2GB)[/dim]")
    
    # 根据不同组合情况处理
    if not current_exists and not global_exists:
        # 情况1: 都不存在 - 启动交互式配置输入
        print(f"\n💡 [bold yellow]未找到任何配置文件，将启动交互式配置向导[/bold yellow]")
        _interactive_config_input(global_env_path)
        
    elif not current_exists and global_exists:
        # 情况2: 只有全局配置存在 - 显示全局配置，询问是否要重新配置
        print(f"\n📋 [bold cyan]检测到全局配置文件，当前配置:[/bold cyan]")
        _display_config_content(global_env_path)
        
        choice = safe_prompt_operation(
            "\n🔧 请选择操作:\n"
            "   1. 保持当前全局配置 (推荐)\n"
            "   2. 重新配置 (会覆盖现有配置)\n"
            "   3. 验证当前配置\n"
            "请输入选择 (1-3)", 
            default="1", 
            show_default=False
        )
        
        if choice == "1":
            print("✅ [bold green]保持当前全局配置[/bold green]")
        elif choice == "2":
            confirm = safe_prompt_operation("⚠️  确认要覆盖现有全局配置吗? (y/N)", default="n", show_default=False).lower()
            if confirm in ['y', 'yes', '是', '确定']:
                _interactive_config_input(global_env_path)
            else:
                print("❌ 重新配置已取消")
        elif choice == "3":
            print("\n🔍 [bold blue]验证全局配置...[/bold blue]")
            validate_existing_config(global_env_path)
        else:
            print("✅ [bold green]保持当前全局配置[/bold green]")
            
    elif current_exists and not global_exists:
        # 情况3: 只有当前目录配置存在 - 显示当前配置，询问是否复制到全局
        print(f"\n📋 [bold cyan]检测到当前目录配置文件，内容如下:[/bold cyan]")
        _display_config_content(current_env_path)
        
        response = safe_prompt_operation("是否将此配置复制到全局配置? (y/N)", default="y", show_default=False).lower()
        
        if response in ['y', 'yes', '是', '确定']:
            _copy_config_with_validation(current_env_path, global_env_path)
        else:
            print("⏭️  跳过复制，仅验证当前目录配置")
            print("\n🔍 [bold blue]验证当前目录配置...[/bold blue]")
            validate_existing_config(current_env_path)
            
    else:
        # 情况4: 两个配置都存在 - 显示两个配置，让用户选择
        print(f"\n📋 [bold cyan]检测到两个配置文件:[/bold cyan]")
        
        print(f"\n🏠 [bold yellow]当前目录配置 (.env):[/bold yellow]")
        _display_config_content(current_env_path)
        
        print(f"\n🌐 [bold yellow]全局配置 ({global_env_path}):[/bold yellow]")
        _display_config_content(global_env_path)
        
        choice = safe_prompt_operation(
            "\n🔧 请选择操作:\n"
            "   1. 保持现有配置不变\n"
            "   2. 用当前目录配置覆盖全局配置\n"
            "   3. 重新配置 (覆盖全局配置)\n"
            "   4. 验证现有配置\n"
            "请输入选择 (1-4)", 
            default="1", 
            show_default=False
        )
        
        if choice == "1":
            print("✅ [bold green]保持现有配置不变[/bold green]")
        elif choice == "2":
            confirm = safe_prompt_operation("⚠️  确认用当前目录配置覆盖全局配置吗? (y/N)", default="n", show_default=False).lower()
            if confirm in ['y', 'yes', '是', '确定']:
                _copy_config_with_validation(current_env_path, global_env_path)
            else:
                print("❌ 覆盖操作已取消")
        elif choice == "3":
            confirm = safe_prompt_operation("⚠️  确认要重新配置并覆盖全局配置吗? (y/N)", default="n", show_default=False).lower()
            if confirm in ['y', 'yes', '是', '确定']:
                _interactive_config_input(global_env_path)
            else:
                print("❌ 重新配置已取消")
        elif choice == "4":
            validate_choice = safe_prompt_operation(
                "验证哪个配置?\n"
                "   1. 当前目录配置\n"
                "   2. 全局配置\n"
                "   3. 两个都验证\n"
                "请输入选择 (1-3)", 
                default="3", 
                show_default=False
            )
            if validate_choice == "1":
                print("\n🔍 [bold blue]验证当前目录配置...[/bold blue]")
                validate_existing_config(current_env_path)
            elif validate_choice == "2":
                print("\n🔍 [bold blue]验证全局配置...[/bold blue]")
                validate_existing_config(global_env_path)
            else:
                print("\n🔍 [bold blue]验证当前目录配置...[/bold blue]")
                validate_existing_config(current_env_path)
                print("\n🔍 [bold blue]验证全局配置...[/bold blue]")
                validate_existing_config(global_env_path)
        else:
            print("✅ [bold green]保持现有配置不变[/bold green]")
    
    # 🆕 统一处理转录模型下载建议 (所有配置情况处理完成后)
    # 检查是否有有效的全局配置
    has_valid_config = global_env_path.exists()
    
    if has_valid_config and not model_available:
        # 有配置且模型未下载 - 提供智能下载建议
        _handle_model_download_suggestion()
    elif not has_valid_config and not model_available:
        # 无配置且模型未下载 - 仅显示下载指南
        _display_model_download_guide()
    
    # 显示最终的系统状态总结
    if has_valid_config:
        _display_system_status_summary()
    
    print("\n🎉 [bold green]配置初始化完成！[/bold green]")


def _display_config_content(env_path: Path):
    """显示配置文件内容（隐藏敏感信息）"""
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析配置信息
        config_info = {}
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                config_info[key] = value
        
        # 分类显示配置
        if 'OPENAI_BASE_URL' in config_info:
            print(f"   🌐 API URL: {config_info['OPENAI_BASE_URL']}")
        
        if 'OPENAI_API_KEY' in config_info:
            api_key = config_info['OPENAI_API_KEY']
            masked_value = api_key[:10] + '*' * (len(api_key) - 10) if len(api_key) > 10 else '*' * len(api_key)
            print(f"   🔑 API Key: {masked_value}")
        
        # 显示模型配置
        model_configs = []
        if 'SPLIT_MODEL' in config_info:
            model_configs.append(f"断句: {config_info['SPLIT_MODEL']}")
        if 'TRANSLATION_MODEL' in config_info:
            model_configs.append(f"翻译: {config_info['TRANSLATION_MODEL']}")
        if 'SUMMARY_MODEL' in config_info:
            model_configs.append(f"总结: {config_info['SUMMARY_MODEL']}")
        if 'LLM_MODEL' in config_info:
            model_configs.append(f"默认: {config_info['LLM_MODEL']}")
        
        if model_configs:
            print("   🤖 模型配置:")
            for model_config in model_configs:
                print(f"      • {model_config}")
        
        # 显示其他配置
        other_configs = []
        for key, value in config_info.items():
            if key not in ['OPENAI_BASE_URL', 'OPENAI_API_KEY', 'SPLIT_MODEL', 'TRANSLATION_MODEL', 'SUMMARY_MODEL', 'LLM_MODEL']:
                if key == 'HF_ENDPOINT' and value:
                    other_configs.append(f"🏗️  HF 镜像站: {value}")
                else:
                    other_configs.append(f"{key}: {value}")
        
        if other_configs:
            print("   ⚙️  其他配置:")
            for other_config in other_configs:
                print(f"      • {other_config}")
                
    except Exception as e:
        print(f"⚠️  读取配置文件失败: {e}")


def _copy_config_with_validation(source_path: Path, target_path: Path):
    """验证并复制配置文件"""
    # 先验证现有配置
    print("\n🔍 [bold blue]正在验证配置...[/bold blue]")
    validation_success = validate_existing_config_and_return_result(source_path)
    
    if not validation_success:
        print("\n⚠️  [bold yellow]配置验证失败，请检查模型名称和网络连接[/bold yellow]")
        continue_response = safe_prompt_operation("是否仍然复制配置? (y/N)", default="n", show_default=False).lower()
        if continue_response not in ['y', 'yes', '是', '确定']:
            print("❌ 配置复制已取消")
            raise typer.Exit(code=1)
    
    # 验证通过后再复制
    try:
        import shutil
        shutil.copy2(source_path, target_path)
        print(f"\n✅ [bold green]配置已保存到:[/bold green] [cyan]{target_path}[/cyan]")
        print("\n🎉 [bold green]配置完成！现在你可以在任意目录下运行 translate 命令！[/bold green]")
        
    except Exception as e:
        print(f"[bold red]❌ 复制失败: {e}[/bold red]")
        raise typer.Exit(code=1)


def _save_config_immediately(global_env_path: Path, base_url: str, api_key: str, 
                           split_model: str, translation_model: str, summary_model: str, 
                           llm_model: str, hf_endpoint: str = None):
    """立即保存配置文件"""
    # 生成配置文件内容
    hf_endpoint_config = f"\n# Hugging Face 镜像站地址 (用于模型下载)\n# 留空使用默认官方地址，设置后可提高国内下载成功率\nHF_ENDPOINT={hf_endpoint or ''}\n" if hf_endpoint else "\n# Hugging Face 镜像站地址 (用于模型下载)\n# 取消注释并设置镜像站可提高国内下载成功率\n# HF_ENDPOINT=https://hf-mirror.com\n"
    
    config_content = f"""# Subtitle Translator 配置文件
# 由 translate init 命令自动生成

# ======== API 配置 ========
# API 基础URL
OPENAI_BASE_URL={base_url}

# API 密钥
OPENAI_API_KEY={api_key}
{hf_endpoint_config}
# ======== 模型配置 ========
# 断句模型 - 负责将长句分割成适合字幕显示的短句
SPLIT_MODEL={split_model}

# 翻译模型 - 负责将字幕翻译成目标语言
TRANSLATION_MODEL={translation_model}

# 总结模型 - 负责分析字幕内容并生成摘要
SUMMARY_MODEL={summary_model}

# 兼容性：默认模型 (如果上述模型未设置，将使用此模型)
LLM_MODEL={llm_model}

# ======== 使用说明 ========
# 1. 你现在可以在任意目录下运行 translate 命令
# 2. 如需修改配置，可以编辑此文件或重新运行 translate init
# 3. 分别配置的模型会优先使用，如未设置则回退到 LLM_MODEL
# 4. HF_ENDPOINT 用于设置 Hugging Face 镜像站，可提高模型下载成功率
"""
    
    # 保存到全局配置
    try:
        with open(global_env_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"\n✅ [bold green]配置已保存到:[/bold green] [cyan]{global_env_path}[/cyan]")
        return True
        
    except Exception as e:
        print(f"[bold red]❌ 保存配置失败: {e}[/bold red]")
        return False


def _execute_predownload():
    """执行转录模型预下载（不显示介绍信息和询问）"""
    print("\n📥 [bold blue]开始预下载默认转录模型...[/bold blue]")
    
    # 🎯 关键修复：重新加载全局配置环境变量
    from dotenv import load_dotenv
    global_env_path = get_global_env_path()
    if global_env_path.exists():
        load_dotenv(global_env_path, override=True)
        print("🔄 [dim]已重新加载全局配置环境变量[/dim]")
    
    # 显示实际使用的下载端点
    import os
    current_hf_endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    if current_hf_endpoint and current_hf_endpoint.strip() and current_hf_endpoint != "https://huggingface.co":
        print(f"🏗️  [cyan]使用配置的镜像站: {current_hf_endpoint}[/cyan]")
    else:
        print("🌐 [dim]使用官方地址下载[/dim]")
    
    try:
        default_model = "mlx-community/parakeet-tdt-0.6b-v2"
        
        # 尝试预下载模型
        from_pretrained(default_model, show_progress=True)
        print("✅ [bold green]默认转录模型预下载成功！[/bold green]")
        return True
        
    except Exception as e:
        print(f"⚠️  [bold yellow]模型预下载失败: {e}[/bold yellow]")
        print("💡 [dim]不用担心，配置已保存成功，首次使用时会自动下载模型[/dim]")
        return False


def _handle_predownload():
    """处理转录模型预下载"""
    print("\n🤖 [bold blue]转录模型预下载[/bold blue]")
    print("字幕翻译工具需要下载转录模型来处理音频文件：")
    print("• 默认模型：mlx-community/parakeet-tdt-0.6b-v2")
    print("• 模型大小：约 1.2GB")
    print("• 首次使用时会自动下载，但可能影响处理速度")
    
    predownload_response = safe_prompt("\n📥 是否现在预下载默认转录模型? (y/N)", default="y", show_default=False).lower()
    should_predownload = predownload_response in ['y', 'yes', '是', '确定']
    
    if should_predownload:
        print("\n📥 [bold blue]开始预下载默认转录模型...[/bold blue]")
        
        # 🎯 关键修复：重新加载全局配置环境变量
        from dotenv import load_dotenv
        global_env_path = get_global_env_path()
        if global_env_path.exists():
            load_dotenv(global_env_path, override=True)
            print("🔄 [dim]已重新加载全局配置环境变量[/dim]")
        
        # 显示实际使用的下载端点
        import os
        current_hf_endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co")
        if current_hf_endpoint and current_hf_endpoint.strip() and current_hf_endpoint != "https://huggingface.co":
            print(f"🏗️  [cyan]使用配置的镜像站: {current_hf_endpoint}[/cyan]")
        else:
            print("🌐 [dim]使用官方地址下载[/dim]")
        
        try:
            default_model = "mlx-community/parakeet-tdt-0.6b-v2"
            
            # 尝试预下载模型
            from_pretrained(default_model, show_progress=True)
            print("✅ [bold green]默认转录模型预下载成功！[/bold green]")
            return True
            
        except Exception as e:
            print(f"⚠️  [bold yellow]模型预下载失败: {e}[/bold yellow]")
            print("💡 [dim]不用担心，配置已保存成功，首次使用时会自动下载模型[/dim]")
            return False
    else:
        print("\n⏭️  [dim]跳过模型预下载，首次使用时会自动下载[/dim]")
        return True


def _display_config_summary(base_url: str, api_key: str, hf_endpoint: str, 
                          split_model: str, translation_model: str, summary_model: str, 
                          llm_model: str, use_separate_models: bool):
    """显示配置摘要"""
    print("\n📋 [bold green]配置摘要:[/bold green]")
    print(f"   🌐 API URL: {base_url}")
    print(f"   🔑 API Key: {api_key[:10]}{'*' * (len(api_key) - 10)}")
    if hf_endpoint:
        print(f"   🏗️  HF 镜像站: [cyan]{hf_endpoint}[/cyan]")
    else:
        print(f"   🏗️  HF 镜像站: [dim]默认官方地址[/dim]")
    if use_separate_models:
        print(f"   🔤 断句模型: [cyan]{split_model}[/cyan]")
        print(f"   🌍 翻译模型: [cyan]{translation_model}[/cyan]")
        print(f"   📊 总结模型: [cyan]{summary_model}[/cyan]")
        print(f"   🤖 默认模型: [cyan]{llm_model}[/cyan]")
    else:
        print(f"   🤖 统一模型: [cyan]{llm_model}[/cyan]")


def _interactive_config_input(global_env_path: Path):
    """交互式输入配置"""
    import sys
    
    # 检查是否在交互式终端中
    if not sys.stdin.isatty():
        print("[bold red]❌ 当前不在交互式终端中，无法进行配置输入[/bold red]")
        print("请在支持交互输入的终端中运行此命令，或手动创建配置文件：")
        print(f"   配置文件路径: [cyan]{global_env_path}[/cyan]")
        print("   可参考项目根目录的 env.example 文件")
        raise typer.Exit(code=1)
    
    base_url = safe_prompt("🌐 API基础URL")
    
    # API密钥
    api_key = safe_prompt("🔑 API密钥")
    
    if not api_key.strip():
        print("[bold red]❌ API密钥不能为空[/bold red]")
        raise typer.Exit(code=1)

    # Hugging Face 镜像站配置
    print("\n🏗️  [bold blue]Hugging Face 下载配置[/bold blue]")
    print("为了提高模型下载成功率，可以配置镜像站：")
    print("• 官方地址：https://huggingface.co (默认)")
    print("• 镜像站：https://hf-mirror.com (推荐国内用户)")
    
    hf_endpoint_response = safe_prompt("🌐 是否使用 Hugging Face 镜像站? (y/N)", default="n", show_default=False).lower()
        
    use_hf_mirror = hf_endpoint_response in ['y', 'yes', '是', '确定']
    
    if use_hf_mirror:
        # 提供几个镜像站选择
        print("\n📋 可选镜像站：")
        print("1. https://hf-mirror.com (推荐)")
        print("2. https://huggingface.co (官方，默认)")
        print("3. 手动输入其他镜像站")
        
        mirror_choice = safe_prompt("请选择镜像站 (1-3)", default="1", show_default=False)
        
        if mirror_choice == "1":
            hf_endpoint = "https://hf-mirror.com"
        elif mirror_choice == "2":
            hf_endpoint = "https://huggingface.co"
        elif mirror_choice == "3":
            hf_endpoint = safe_prompt("请输入镜像站地址")
        else:
            hf_endpoint = "https://hf-mirror.com"  # 默认使用推荐镜像站
    else:
        hf_endpoint = None

    print("\n🤖 [bold blue]模型配置[/bold blue]")
    print("字幕翻译工具支持为不同功能使用不同的模型：")
    print("• 断句模型：将长句分割成适合字幕显示的短句")
    print("• 翻译模型：翻译字幕内容")
    print("• 总结模型：分析字幕内容并生成摘要")
    
    # 询问是否要分别配置模型
    separate_models_response = safe_prompt("\n🔧 是否为不同功能分别配置模型? (y/N)", default="y", show_default=False).lower()
    use_separate_models = separate_models_response in ['y', 'yes', '是', '确定']
    
    if use_separate_models:
        # 分别配置三个模型
        print("\n🔤 [bold yellow]断句模型配置[/bold yellow]")
        split_model = safe_prompt("断句模型")
        while not split_model.strip():
            print("❌ 断句模型不能为空")
            split_model = safe_prompt("断句模型")
        
        print("\n🌍 [bold yellow]翻译模型配置[/bold yellow]")
        translation_model = safe_prompt("翻译模型")
        while not translation_model.strip():
            print("❌ 翻译模型不能为空")
            translation_model = safe_prompt("翻译模型")
        
        print("\n📊 [bold yellow]总结模型配置[/bold yellow]")
        summary_model = safe_prompt("总结模型")
        while not summary_model.strip():
            print("❌ 总结模型不能为空")
            summary_model = safe_prompt("总结模型")
        
        # 兼容性默认模型
        llm_model = split_model
    else:
        print("\n🤖 [bold yellow]统一模型配置[/bold yellow]")
        llm_model = safe_prompt("LLM模型")
        while not llm_model.strip():
            print("❌ LLM模型不能为空")
            llm_model = safe_prompt("LLM模型")
        
        # 统一使用一个模型
        split_model = llm_model
        translation_model = llm_model
        summary_model = llm_model
    
    # 验证配置
    print("\n🔍 [bold blue]正在验证 API 配置...[/bold blue]")
    
    # 获取所有需要验证的模型
    unique_models = {}
    unique_models['断句模型'] = split_model
    unique_models['翻译模型'] = translation_model
    unique_models['总结模型'] = summary_model
    unique_models['默认模型'] = llm_model
        
    # 去重：只测试不同的模型
    tested_models = set()
    validation_results = []
    
    for model_type, model_name in unique_models.items():
        if model_name not in tested_models:
            print(f"🔌 测试 {model_name}...")
            success, message = test_openai(base_url, api_key, model_name)
            tested_models.add(model_name)
            
            validation_results.append({
                'model': model_name,
                'success': success,
                'message': message,
                'types': [model_type]
            })
        else:
            # 如果模型已经测试过，找到之前的结果并添加类型
            for result in validation_results:
                if result['model'] == model_name:
                    result['types'].append(model_type)
                    break
    
    # 显示验证结果
    print("\n📊 [bold blue]验证结果:[/bold blue]")
    all_success = True
    
    for result in validation_results:
        model_types = '、'.join(result['types'])
        if result['success']:
            print(f"   ✅ {result['model']} ({model_types})")
            print(f"      响应: {result['message'][:60]}...")
        else:
            print(f"   ❌ {result['model']} ({model_types})")
            print(f"      错误: {result['message']}")
            all_success = False
    
    if not all_success:
        print("\n⚠️  [bold yellow]部分模型验证失败，请检查模型名称和网络连接[/bold yellow]")
        continue_response = safe_prompt("是否继续保存配置? (y/N)", default="n", show_default=False).lower()
        if continue_response not in ['y', 'yes', '是', '确定']:
            print("❌ 配置保存已取消")
            raise typer.Exit(code=1)
    
    # 🎯 关键改进：API验证通过后立即保存配置
    print("\n💾 [bold blue]保存配置文件...[/bold blue]")
    config_saved = _save_config_immediately(global_env_path, base_url, api_key, 
                                           split_model, translation_model, summary_model, 
                                           llm_model, hf_endpoint)
    
    if not config_saved:
        print("[bold red]❌ 配置保存失败，无法继续[/bold red]")
        raise typer.Exit(code=1)
    
    # 显示配置摘要
    _display_config_summary(base_url, api_key, hf_endpoint, split_model, 
                           translation_model, summary_model, llm_model, use_separate_models)
    
    # 处理转录模型预下载 (独立步骤，失败不影响配置)
    print("\n" + "="*50)
    predownload_success = _handle_predownload()
    
    # 最终状态提示
    if predownload_success:
        print("\n🎉 [bold green]配置完成且模型预下载成功！现在你可以在任意目录下运行 translate 命令！[/bold green]")
    else:
        print("\n🎉 [bold green]配置已完成！现在你可以在任意目录下运行 translate 命令！[/bold green]")
        print("💡 [dim]转录模型将在首次使用时自动下载[/dim]") 