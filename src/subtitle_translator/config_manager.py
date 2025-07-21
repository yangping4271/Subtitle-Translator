"""
配置管理模块 - 配置验证和初始化
"""
import os
from pathlib import Path

import typer
from rich import print

from .env_setup import setup_environment, get_app_config_dir, get_global_env_path
from .translation_core.utils.test_openai import test_openai


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


def init_config():
    """初始化全局配置 - 检查当前目录.env文件或交互式输入配置"""
    print("[bold green]🚀 字幕翻译工具配置初始化[/bold green]")
    
    # 获取全局配置目录和文件路径
    app_dir = get_app_config_dir()
    global_env_path = get_global_env_path()
    current_env_path = Path(".env")
    
    # 确保全局配置目录存在
    app_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查当前目录是否有.env文件
    if current_env_path.exists():
        # 显示当前.env文件内容（隐藏敏感信息）
        print("\n📋 [bold cyan]当前配置文件内容:[/bold cyan]")
        try:
            with open(current_env_path, 'r', encoding='utf-8') as f:
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
        
        # 询问是否复制
        response = typer.prompt("是否将此配置复制到全局配置? (y/N)", default="n", show_default=False).lower()
        
        if response in ['y', 'yes', '是', '确定']:
            # 先验证现有配置
            print("\n🔍 [bold blue]正在验证现有配置...[/bold blue]")
            validation_success = validate_existing_config_and_return_result(current_env_path)
            
            if not validation_success:
                print("\n⚠️  [bold yellow]配置验证失败，请检查模型名称和网络连接[/bold yellow]")
                continue_response = typer.prompt("是否仍然复制配置? (y/N)", default="n", show_default=False).lower()
                if continue_response not in ['y', 'yes', '是', '确定']:
                    print("❌ 配置复制已取消")
                    raise typer.Exit(code=1)
            
            # 验证通过后再复制
            try:
                import shutil
                shutil.copy2(current_env_path, global_env_path)
                print(f"\n✅ [bold green]配置已保存到:[/bold green] [cyan]{global_env_path}[/cyan]")
                print("\n🎉 [bold green]配置完成！现在你可以在任意目录下运行 translate 命令！[/bold green]")
                
            except Exception as e:
                print(f"[bold red]❌ 复制失败: {e}[/bold red]")
                raise typer.Exit(code=1)
        else:
            print("⏭️  跳过复制，配置未更改")
            
            # 即使不复制，也验证当前配置
            print("\n🔍 [bold blue]验证当前目录的配置...[/bold blue]")
            validate_existing_config(current_env_path)
    
    else:
        # 交互式输入配置
        _interactive_config_input(global_env_path)


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
    
    try:
        base_url = typer.prompt("🌐 API基础URL")
    except (KeyboardInterrupt, typer.Abort):
        print("\n❌ 配置已取消")
        raise typer.Exit(code=1)
    
    # API密钥
    api_key = typer.prompt("🔑 API密钥")
    
    if not api_key.strip():
        print("[bold red]❌ API密钥不能为空[/bold red]")
        raise typer.Exit(code=1)

    # Hugging Face 镜像站配置
    print("\n🏗️  [bold blue]Hugging Face 下载配置[/bold blue]")
    print("为了提高模型下载成功率，可以配置镜像站：")
    print("• 官方地址：https://huggingface.co (默认)")
    print("• 镜像站：https://hf-mirror.com (推荐国内用户)")
    
    hf_endpoint_response = typer.prompt("🌐 是否使用 Hugging Face 镜像站? (y/N)", default="n", show_default=False).lower()
    use_hf_mirror = hf_endpoint_response in ['y', 'yes', '是', '确定']
    
    if use_hf_mirror:
        # 提供几个镜像站选择
        print("\n📋 可选镜像站：")
        print("1. https://hf-mirror.com (推荐)")
        print("2. https://huggingface.co (官方，默认)")
        print("3. 手动输入其他镜像站")
        
        mirror_choice = typer.prompt("请选择镜像站 (1-3)", default="1", show_default=False)
        
        if mirror_choice == "1":
            hf_endpoint = "https://hf-mirror.com"
        elif mirror_choice == "2":
            hf_endpoint = "https://huggingface.co"
        elif mirror_choice == "3":
            hf_endpoint = typer.prompt("请输入镜像站地址")
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
    separate_models_response = typer.prompt("\n🔧 是否为不同功能分别配置模型? (y/N)", default="y", show_default=False).lower()
    use_separate_models = separate_models_response in ['y', 'yes', '是', '确定']
    
    if use_separate_models:
        # 分别配置三个模型
        print("\n🔤 [bold yellow]断句模型配置[/bold yellow]")
        split_model = typer.prompt("断句模型")
        while not split_model.strip():
            print("❌ 断句模型不能为空")
            split_model = typer.prompt("断句模型")
        
        print("\n🌍 [bold yellow]翻译模型配置[/bold yellow]")
        translation_model = typer.prompt("翻译模型")
        while not translation_model.strip():
            print("❌ 翻译模型不能为空")
            translation_model = typer.prompt("翻译模型")
        
        print("\n📊 [bold yellow]总结模型配置[/bold yellow]")
        summary_model = typer.prompt("总结模型")
        while not summary_model.strip():
            print("❌ 总结模型不能为空")
            summary_model = typer.prompt("总结模型")
        
        # 兼容性默认模型
        llm_model = split_model
    else:
        print("\n🤖 [bold yellow]统一模型配置[/bold yellow]")
        llm_model = typer.prompt("LLM模型")
        while not llm_model.strip():
            print("❌ LLM模型不能为空")
            llm_model = typer.prompt("LLM模型")
        
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
        continue_response = typer.prompt("是否继续保存配置? (y/N)", default="n", show_default=False).lower()
        if continue_response not in ['y', 'yes', '是', '确定']:
            print("❌ 配置保存已取消")
            raise typer.Exit(code=1)
    
    # API验证通过后，生成配置文件内容
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
        
        # 显示配置摘要
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
        
        print("\n🎉 [bold green]配置完成！现在你可以在任意目录下运行 translate 命令！[/bold green]")
        
    except Exception as e:
        print(f"[bold red]❌ 保存配置失败: {e}[/bold red]")
        raise typer.Exit(code=1) 