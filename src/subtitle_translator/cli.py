import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

import typer
from typing_extensions import Annotated
import glob


# 应用名称，用于配置文件目录
APP_NAME = "subtitle_translator"

# 全局变量，用于跟踪环境是否已经加载
_env_loaded = False

def setup_environment():
    """
    智能加载 .env 文件，解决在不同目录下运行命令的环境变量问题。
    加载顺序 (后者覆盖前者):
    1. 用户全局配置文件 (~/.config/subtitle_translator/.env)
    2. 项目配置文件 (从当前目录向上找到的第一个 .env)
    
    特殊功能：
    - 如果全局配置不存在，但找到项目配置，会自动复制项目配置作为全局配置
    - 使用标准的 .config 目录存储全局配置
    """
    global _env_loaded, logger
    
    # 如果已经加载过环境配置，直接返回
    if _env_loaded:
        return
    
    env_loaded = False
    
    # 准备路径 - 使用标准的 .config 目录
    app_dir = Path.home() / ".config" / APP_NAME
    user_env_path = app_dir / ".env"
    
    # 确保目录存在
    app_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找项目本地的 .env 文件
    project_env_path_str = find_dotenv(usecwd=True)
    project_env_path = Path(project_env_path_str) if project_env_path_str else None
    
    # 🎯 智能配置复制：如果全局配置不存在但项目配置存在，自动复制
    config_copied = False
    if not user_env_path.is_file() and project_env_path and project_env_path.is_file():
        try:
            import shutil
            shutil.copy2(project_env_path, user_env_path)
            config_copied = True
        except Exception as e:
            print(f"⚠️  复制配置文件失败: {e}")

    # 1. 加载用户全局配置文件 (适用于已安装的应用)
    if user_env_path.is_file():
        # 加载全局配置，但不覆盖已存在的环境变量，关闭verbose输出
        load_dotenv(user_env_path, verbose=False)
        env_loaded = True
        
    # 2. 加载项目本地的 .env 文件 (方便开发，并可覆盖全局配置)
    if project_env_path and project_env_path.is_file():
        # 使用 override=True 来覆盖任何已存在的环境变量，确保项目配置优先，关闭verbose输出
        load_dotenv(project_env_path, verbose=False, override=True)
        env_loaded = True
    
    # 标记环境已加载
    _env_loaded = True
    
    # 初始化logger（需要在环境变量加载后进行）
    if logger is None:
        # 检测debug模式：检查命令行参数和环境变量
        import sys
        debug_mode = ('-d' in sys.argv or '--debug' in sys.argv or 
                     os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes'))
        
        from .translation_core.utils.logger import setup_logger
        logger = setup_logger(__name__, debug_mode=debug_mode)
        
        # 只在需要提醒用户或出现问题时输出日志信息
        if config_copied:
            logger.info(f"✅ 首次运行检测到项目配置文件，已自动复制到全局配置:")
            logger.info(f"   源文件: {project_env_path}")
            logger.info(f"   目标文件: {user_env_path}")
            logger.info(f"   现在你可以在任意目录下运行 subtitle-translate 命令！")
        elif not env_loaded:
            logger.warning(
                f"未找到任何 .env 文件。程序将依赖于系统环境变量。\n"
                f"如需通过文件配置，请在项目根目录或用户配置目录 "
                f"({app_dir}) 中创建一个 .env 文件。"
            )
            
            # 检查关键环境变量是否存在
            required_vars = ['OPENAI_BASE_URL', 'OPENAI_API_KEY', 'LLM_MODEL']
            missing_vars = []
            for var in required_vars:
                if not os.environ.get(var):
                    missing_vars.append(var)
            
            if missing_vars:
                logger.error(f"缺少必需的环境变量: {', '.join(missing_vars)}")
                logger.error("请运行 'subtitle-translate init' 来配置API密钥，或设置相应的环境变量。")
                import sys
                sys.exit(1)

# 在所有其他项目导入之前，首先加载环境变量
# setup_environment()  <-- 我将删除这一行


from typing import Optional

from rich import print
import logging

# 导入转录核心
from .transcription_core import from_pretrained
from .transcription_core.cli import to_srt

# 导入翻译核心
from .translation_core.optimizer import SubtitleOptimizer
from .translation_core.summarizer import SubtitleSummarizer
from .translation_core.spliter import merge_segments
from .translation_core.config import get_default_config
from .translation_core.data import load_subtitle, SubtitleData
from .translation_core.utils.test_openai import test_openai
from .translation_core.utils.logger import setup_logger


# 配置日志
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') # 这一行已被移除
# 延迟初始化logger，在setup_environment中初始化
logger = None

class OpenAIAPIError(Exception):
    """OpenAI API 相关错误"""
    pass

class SubtitleTranslatorService:
    def __init__(self):
        self.config = get_default_config()
        self.summarizer = SubtitleSummarizer(config=self.config)

    def _init_translation_env(self, llm_model: str) -> None:
        """初始化翻译环境"""
        if llm_model:
            self.config.llm_model = llm_model

        logger.info(f"使用 {self.config.openai_base_url} 作为API端点")
        logger.info(f"使用 {self.config.llm_model} 作为LLM模型")
        
        success, error_msg = test_openai(self.config.openai_base_url, self.config.openai_api_key, self.config.llm_model)
        if not success:
            raise OpenAIAPIError(error_msg)

    def translate_srt(self, input_srt_path: Path, target_lang: str, output_dir: Path, 
                      llm_model: Optional[str] = None, reflect: bool = False) -> Path:
        """翻译字幕文件"""
        try:
            logger.info("字幕翻译任务开始...")     
            # 初始化翻译环境
            self._init_translation_env(llm_model)
            
            # 加载字幕文件
            asr_data = load_subtitle(str(input_srt_path))
            logger.debug(f"字幕内容: {asr_data.to_txt()[:100]}...")  
            
            # 检查是否需要重新断句 (这里简化处理，如果需要更复杂的断句逻辑，可以从原项目复制)
            if asr_data.is_word_timestamp():
                model = os.getenv("LLM_MODEL")
                logger.info(f"正在使用{model} 断句")
                logger.info(f"句子限制长度为{self.config.max_word_count_english}字")
                asr_data = merge_segments(asr_data, model=model, 
                                       num_threads=self.config.thread_num, 
                                       save_split=None) # 暂时不保存断句结果
            
            # 获取字幕摘要
            summarize_result = self._get_subtitle_summary(asr_data, str(input_srt_path))
            
            # 翻译字幕
            translate_result = self._translate_subtitles(asr_data, summarize_result, reflect)
            
            # 保存字幕
            base_name = input_srt_path.stem
            # 假设目标语言是中文，输出文件名为 original_filename.zh.srt
            # 如果需要支持多种目标语言，这里需要更复杂的逻辑来生成文件名
            zh_output_path = output_dir / f"{base_name}.{target_lang}.srt"
            en_output_path = output_dir / f"{base_name}.en.srt" # 假设也保存英文原版

            asr_data.save_translations_to_files(
                translate_result,
                str(en_output_path),
                str(zh_output_path)
            )
            logger.info(f"翻译完成，输出文件: {zh_output_path}")
            return zh_output_path
                
        except OpenAIAPIError as e:
            error_msg = f"\n{'='*50}\n错误: {str(e)}\n{'='*50}\n"
            logger.error(error_msg)
            raise typer.Exit(code=1)
            
        except Exception as e:
            error_msg = f"\n{'='*50}\n处理过程中发生错误: {str(e)}\n{'='*50}\n"
            logger.exception(error_msg)
            raise typer.Exit(code=1)

    def _get_subtitle_summary(self, asr_data: SubtitleData, input_file: str) -> dict:
        """获取字幕内容摘要"""
        logger.info(f"正在使用 {self.config.llm_model} 总结字幕...")
        summarize_result = self.summarizer.summarize(asr_data.to_txt(), input_file)
        logger.info(f"总结字幕内容:\n{summarize_result.get('summary')}\n")
        return summarize_result

    def _translate_subtitles(self, asr_data: SubtitleData, summarize_result: dict, reflect: bool = False) -> list:
        """翻译字幕内容"""
        logger.info(f"正在使用 {self.config.llm_model} 翻译字幕...")
        try:
            translator = SubtitleOptimizer(
                config=self.config,
                need_reflect=reflect
            )
            translate_result = translator.translate(asr_data, summarize_result)
            return translate_result
        except Exception as e:
            logger.error(f"翻译失败: {str(e)}")
            raise

app = typer.Typer(
    help="一个集成了语音转录、字幕翻译和格式转换的命令行工具",
    epilog="💡 首次使用请运行: subtitle-translate init 来配置API密钥"
)

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_file: Optional[Path] = typer.Option(None, "--input-file", "-i", help="要处理的单个文件路径，如不指定则批量处理当前目录。", exists=True, file_okay=True, dir_okay=False, readable=True),
    max_count: int = typer.Option(-1, "--count", "-n", help="最大处理文件数量，-1表示处理所有文件。"),
    target_lang: str = typer.Option("zh", "--target_lang", "-t", help="目标翻译语言，例如 'zh' (中文), 'en' (英文)。"),
    output_dir: Optional[Path] = typer.Option(None, "--output_dir", "-o", help="输出文件的目录，默认为当前目录。"),
    model: str = typer.Option("mlx-community/parakeet-tdt-0.6b-v2", "--model", help="用于转录的 Parakeet MLX 模型。"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", "-m", help="用于翻译的LLM模型，默认使用配置文件中的设置。"),
    reflect: bool = typer.Option(False, "--reflect", "-r", help="启用反思翻译模式，提高翻译质量但会增加处理时间。"),
    debug: bool = typer.Option(False, "--debug", "-d", help="启用调试日志级别，显示更详细的处理信息。"),
):
    """字幕翻译工具主命令"""
    setup_environment()
    
    # 如果调用了子命令，就不执行主逻辑
    if ctx.invoked_subcommand is not None:
        return


        
    

    # 如果没有指定输出目录，默认使用当前目录
    if output_dir is None:
        output_dir = Path.cwd()
    
    # 确保使用绝对路径，避免相对路径在不同工作目录下的问题
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取要处理的文件列表
    if input_file:
        # 单文件模式
        files_to_process = [input_file]
        print(f"开始处理单个文件: [bold cyan]{input_file.name}[/bold cyan]")
    else:
        # 批量处理模式：查找当前目录中的媒体文件
        import re
        
        MEDIA_EXTENSIONS = ["*.srt", "*.mp3", "*.mp4"]

        # 查找所有媒体文件
        media_files = []
        for pattern in MEDIA_EXTENSIONS:
            media_files.extend(glob.glob(pattern))
        
        if not media_files:
            print("[bold red]当前目录没有找到需要处理的文件 (*.srt, *.mp3, *.mp4)。[/bold red]")
            raise typer.Exit(code=1)
        
        # 提取基础文件名并去重排序
        base_names = set()
        for file in media_files:
            # 移除扩展名和语言后缀
            base_name = re.sub(r'\.(srt|mp3|mp4)$', '', file)
            base_name = re.sub(r'_(en|zh)$', '', base_name)
            base_names.add(base_name)
        
        base_names = sorted(base_names)
        
        # 为每个基础名称找到对应的输入文件
        files_to_process = []
        for base_name in base_names:
            # 跳过已存在.ass文件的
            ass_file = Path(f"{base_name}.ass")
            if ass_file.exists():
                print(f"INFO: {base_name}.ass 已存在，跳过处理。")
                continue
            
            # 确定输入文件优先级：srt > mp3 > mp4
            input_file_found = None
            for ext in ['.srt', '.mp3', '.mp4']:
                candidate = Path(f"{base_name}{ext}")
                if candidate.exists():
                    input_file_found = candidate
                    break
            
            if input_file_found:
                files_to_process.append(input_file_found)
                print(f"INFO: 发现文件 {input_file_found}")
            else:
                print(f"ERROR: 没有找到 {base_name} 的输入文件")
        
        if not files_to_process:
            print("[bold yellow]没有找到需要处理的新文件。[/bold yellow]")
            raise typer.Exit(code=0)
        
        # 应用数量限制
        if max_count > 0:
            files_to_process = files_to_process[:max_count]
        
        print(f"[bold green]开始批量翻译处理，共{len(files_to_process)}个文件...[/bold green]")
        if llm_model:
            print(f"使用LLM模型: [bold cyan]{llm_model}[/bold cyan]")

    # 处理文件
    count = 0
    for i, current_input_file in enumerate(files_to_process):
        print("=" * 50)
        print(f"处理文件 ({i+1}/{len(files_to_process)}): [bold cyan]{current_input_file.name}[/bold cyan]")
        
        try:
            _process_single_file(
                current_input_file, target_lang, output_dir, model, 
                llm_model, reflect, debug
            )
            count += 1
            print(f"[bold green]SUCCESS: {current_input_file.stem} 处理完成！[/bold green]")
            
            # 检查是否生成了ASS文件
            ass_file = output_dir / f"{current_input_file.stem}.ass"
            if ass_file.exists():
                print(f"INFO: 双语ASS文件已生成: {ass_file.name}")
        
        except Exception as e:
            print(f"[bold red]ERROR: {current_input_file.stem} 处理失败！{e}[/bold red]")
        
        print()  # 添加空行分隔
    
    # 显示处理结果
    print("=" * 50)
    print(f"[bold green]批量处理完成！[/bold green]")
    print(f"总计处理文件数: [bold cyan]{count}[/bold cyan]")
    
    if count > 0:
        print("\n生成的文件：")
        ass_files = list(output_dir.glob("*.ass"))
        if ass_files:
            for f in ass_files:
                print(f"  {f.name}")
        else:
            print("  没有生成ASS文件")
        
        print("\n原始字幕文件：")
        srt_files = [f for f in output_dir.glob("*.srt") if not ("_zh" in f.name or "_en" in f.name)]
        if srt_files:
            for f in srt_files:
                print(f"  {f.name}")
        else:
            print("  没有保留的SRT文件")
    
    print("处理完毕！")


def _process_single_file(
    input_file: Path, target_lang: str, output_dir: Path, 
    model: str, llm_model: Optional[str], reflect: bool, debug: bool
):
    """处理单个文件的核心逻辑"""

    # 检测输入文件类型
    if input_file.suffix.lower() == '.srt':
        print("[bold yellow]>>> 检测到SRT文件，跳过转录步骤...[/bold yellow]")
        temp_srt_path = input_file
    else:
        # --- 转录阶段 ---
        print("[bold green]>>> 开始转录...[/bold green]")
        temp_srt_path = output_dir / f"{input_file.stem}.srt"
        try:
            # 模拟 parakeet-mlx 的转录过程
            # 实际这里需要调用 parakeet-mlx 的核心转录函数
            # 由于 parakeet-mlx 的 cli.py 中的 main 函数直接处理文件并保存，
            # 我们需要将其核心逻辑提取出来，或者直接调用其内部的 transcribe 方法。
            # 这里暂时用一个占位符，后续需要将 parakeet-mlx 的转录逻辑封装成一个可调用的函数。
            
            # 假设 from_pretrained 返回一个模型实例，并且该实例有 transcribe 方法
            # 并且 transcribe 方法返回 AlignedResult
            loaded_model = from_pretrained(model)
            
            # 对于大文件，使用分块处理避免内存溢出
            # 使用与原始parakeet-mlx相同的默认值：120秒分块，15秒重叠
            result = loaded_model.transcribe(input_file, chunk_duration=120.0, overlap_duration=15.0)
            
            # 将转录结果保存为 SRT，使用 timestamps=True 获得更精细的时间戳
            srt_content = to_srt(result, timestamps=True)
            with open(temp_srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            print(f"[bold green]转录完成，SRT文件保存至:[/bold green] {temp_srt_path}")

        except Exception as e:
            print(f"[bold red]转录失败:[/bold red] {e}")
            raise typer.Exit(code=1)

    final_translated_zh_path = None
    final_translated_en_path = None

    # --- 翻译阶段 ---
    print("[bold green]>>> 开始翻译...[/bold green]")
    try:
        translator_service = SubtitleTranslatorService()
    except Exception as init_error:
        print(f"[bold red]创建翻译服务失败:[/bold red] {init_error}")
        raise
    try:
        final_translated_zh_path = translator_service.translate_srt(
            input_srt_path=temp_srt_path,
            target_lang=target_lang,
            output_dir=output_dir,
            llm_model=llm_model,
            reflect=reflect
        )
        # 确保这里正确赋值
        final_translated_en_path = output_dir / f"{temp_srt_path.stem}.en.srt"

        print(f"[bold green]翻译完成，中文翻译文件保存至:[/bold green] {final_translated_zh_path}")
        print(f"[bold green]英文翻译文件保存至:[/bold green] {final_translated_en_path}")

        # --- 转换为 ASS ---
        print("[bold green]>>> 正在转换为 ASS 格式...[/bold green]")

        # 提取 srt2ass.py 的核心逻辑
        from .translation_core.utils.ass_converter import convert_srt_to_ass

        final_ass_path = convert_srt_to_ass(final_translated_zh_path, final_translated_en_path, output_dir)
        print(f"[bold green]ASS 文件生成成功:[/bold green] {final_ass_path}")

    except Exception as e:
        print(f"[bold red]翻译或 ASS 转换失败:[/bold red] {e}")
        raise typer.Exit(code=1)
    finally:
        # --- 清理中间翻译文件，保留原始转录文件 ---
        print("[bold green]>>> 正在清理中间翻译文件...[/bold green]")
        if final_translated_zh_path and final_translated_zh_path.exists():
            os.remove(final_translated_zh_path)
            print(f"已删除中间文件: {final_translated_zh_path}")
        if final_translated_en_path and final_translated_en_path.exists():
            os.remove(final_translated_en_path)
            print(f"已删除中间文件: {final_translated_en_path}")
        
        # 处理原始SRT文件
        if temp_srt_path and temp_srt_path.exists():
            if input_file.suffix.lower() == '.srt':
                print(f"[bold green]输入文件为SRT，保持原文件不变:[/bold green] {temp_srt_path}")
            else:
                print(f"[bold green]保留原始转录文件:[/bold green] {temp_srt_path}")

@app.command("init")
def init():
    """初始化全局配置 - 检查当前目录.env文件或交互式输入配置"""
    print("[bold green]🚀 字幕翻译工具配置初始化[/bold green]")
    
    # 获取全局配置目录和文件路径 - 使用标准的 .config 目录
    app_dir = Path.home() / ".config" / APP_NAME
    global_env_path = app_dir / ".env"
    current_env_path = Path(".env")
    
    # 确保全局配置目录存在
    app_dir.mkdir(parents=True, exist_ok=True)
    
    
    
    # 检查当前目录是否有.env文件
    if current_env_path.exists():
        
        
        # 显示当前.env文件内容（隐藏敏感信息）
        try:
            with open(current_env_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            
            for line in content.split('\n'):
                if line.strip() and not line.strip().startswith('#'):
                    if 'API_KEY' in line:
                        key, value = line.split('=', 1)
                        masked_value = value[:10] + '*' * (len(value) - 10) if len(value) > 10 else '*' * len(value)
                        print(f"   {key}={masked_value}")
                    else:
                        print(f"   {line}")
        except Exception as e:
            print(f"⚠️  读取配置文件失败: {e}")
        
        # 询问是否复制
        
        
        # 使用标准输入读取用户选择
        response = typer.prompt("是否将此配置复制到全局配置? (y/N)", default="n", show_default=False).lower()
        
        if response in ['y', 'yes', '是', '确定']:
            try:
                import shutil
                shutil.copy2(current_env_path, global_env_path)
                print(f"✅ 配置已复制到: [bold green]{global_env_path}[/bold green]")
                print("🎉 现在你可以在任意目录下运行 subtitle-translate 命令！")
            except Exception as e:
                print(f"[bold red]❌ 复制失败: {e}[/bold red]")
                raise typer.Exit(code=1)
        else:
            print("⏭️  跳过复制，配置未更改")
    
    else:
        
        
        # 交互式输入配置
        
        base_url = typer.prompt("🌐 API基础URL", default="https://api.openai.com/v1")
        
        # API密钥
        api_key = typer.prompt("🔑 API密钥")
        
        if not api_key.strip():
            print("[bold red]❌ API密钥不能为空[/bold red]")
            raise typer.Exit(code=1)
        
        # LLM模型
        model_options = [
            "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo",
            "claude-3-sonnet", "claude-3-haiku",
            "google/gemini-2.5-flash-lite-preview-06-17"
        ]
        
        
        
        llm_model = typer.prompt("请选择LLM模型 (输入序号或直接输入模型名)", default="gpt-4o-mini")
        
        # 如果输入的是数字，转换为对应的模型
        if llm_model.isdigit():
            idx = int(llm_model) - 1
            if 0 <= idx < len(model_options):
                llm_model = model_options[idx]
            else:
                print("⚠️  无效选择，使用默认模型: gpt-4o-mini")
                llm_model = "gpt-4o-mini"
        
        # 可选配置
        log_level = typer.prompt("📊 日志级别 (DEBUG/INFO/WARNING/ERROR)", default="INFO").upper()
        
        debug_response = typer.prompt("🐛 启用调试模式? (y/N)", default="n", show_default=False).lower()
        debug_mode = debug_response in ['y', 'yes', '是', '确定']
        
        # 生成配置文件内容
        config_content = f"""# Subtitle Translator 配置文件
# 由 subtitle-translate init 命令自动生成

# OpenAI API 配置 (必需)
# API 基础URL
OPENAI_BASE_URL={base_url}

# API 密钥
OPENAI_API_KEY={api_key}

# 默认 LLM 模型
LLM_MODEL={llm_model}

# 可选配置
# 日志级别
LOG_LEVEL={log_level}

# 调试模式
DEBUG={str(debug_mode).lower()}

# 使用说明
# 1. 此配置文件已保存到全局配置目录 (~/.config/subtitle_translator/.env)
# 2. 你现在可以在任意目录下运行 subtitle-translate 命令
# 3. 如需修改配置，可以编辑此文件或重新运行 subtitle-translate init
"""
        
        # 保存到全局配置
        try:
            with open(global_env_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            print(f"\n✅ 配置已保存到: [bold green]{global_env_path}[/bold green]")
            
            # 显示配置摘要
            
            print(f"   🌐 API URL: {base_url}")
            print(f"   🔑 API Key: {api_key[:10]}{'*' * (len(api_key) - 10)}")
            print(f"   🤖 LLM模型: {llm_model}")
            print(f"   📊 日志级别: {log_level}")
            print(f"   🐛 调试模式: {debug_mode}")
            
            print("\n🎉 配置完成！现在你可以在任意目录下运行 subtitle-translate 命令！")
            
        except Exception as e:
            print(f"[bold red]❌ 保存配置失败: {e}[/bold red]")
            raise typer.Exit(code=1)
    
    # 验证配置
    
    try:
        # 重新加载环境变量
        global _env_loaded
        _env_loaded = False
        setup_environment()
        
        # 测试API连接
        from .translation_core.utils.test_openai import test_openai
        
        base_url = os.getenv('OPENAI_BASE_URL')
        api_key = os.getenv('OPENAI_API_KEY')
        model = os.getenv('LLM_MODEL')
        
        
        success, message = test_openai(base_url, api_key, model)
        
        if success:
            print("✅ API连接测试成功！")
            print(f"响应: {message[:100]}...")
        else:
            print(f"❌ API连接测试失败: {message}")
            
    except Exception as e:
        print(f"⚠️  配置验证过程中出现错误: {e}")
        print("但配置文件已成功保存，你可以稍后手动验证")

if __name__ == "__main__":
    app()
