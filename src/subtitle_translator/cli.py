import typer
from typing_extensions import Annotated
from pathlib import Path
import logging
import glob
import os
from dotenv import load_dotenv, find_dotenv

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
    """
    global _env_loaded
    
    # 如果已经加载过环境配置，直接返回
    if _env_loaded:
        return
    
    env_loaded = False
    
    # 1. 加载用户全局配置文件 (适用于已安装的应用)
    # typer.get_app_dir() 会创建目录，如果它不存在的话
    app_dir = Path(typer.get_app_dir(APP_NAME, force_posix=True))
    user_env_path = app_dir / ".env"
    
    # 确保目录存在
    app_dir.mkdir(parents=True, exist_ok=True)

    if user_env_path.is_file():
        # 加载全局配置，但不覆盖已存在的环境变量，关闭verbose输出
        load_dotenv(user_env_path, verbose=False)
        logging.info(f"已加载用户全局环境配置: {user_env_path}")
        env_loaded = True
        
    # 2. 加载项目本地的 .env 文件 (方便开发，并可覆盖全局配置)
    # find_dotenv(usecwd=True) 会从当前工作目录向上查找
    project_env_path_str = find_dotenv(usecwd=True)
    if project_env_path_str:
        project_env_path = Path(project_env_path_str)
        if project_env_path.is_file():
            # 使用 override=True 来覆盖任何已存在的环境变量，确保项目配置优先，关闭verbose输出
            load_dotenv(project_env_path, verbose=False, override=True)
            logging.info(f"已加载项目环境配置 (覆盖全局配置): {project_env_path}")
            env_loaded = True

    if not env_loaded:
        logging.warning(
            f"未找到任何 .env 文件。程序将依赖于系统环境变量。\n"
            f"如需通过文件配置，请在项目根目录或用户配置目录 "
            f"({app_dir}) 中创建一个 .env 文件。"
        )
    
    # 标记环境已加载
    _env_loaded = True

# 在所有其他项目导入之前，首先加载环境变量
# setup_environment()  <-- 我将删除这一行

logging.info(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'NOT_SET')}")

import sys
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
from .translation_core.utils.test_opanai import test_openai
from .translation_core.utils.logger import setup_logger
# from .translation_core import translate_and_convert  # 这个函数不存在，暂时注释掉

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

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

app = typer.Typer(help="一个集成了语音转录、字幕翻译和格式转换的命令行工具")

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
    # 将环境设置移到这里，确保只执行一次
    setup_environment()

    if ctx.invoked_subcommand is not None:
        return
        
    if debug:
        os.environ['DEBUG'] = 'true'
        logger.setLevel(os.environ.get('LOG_LEVEL', 'DEBUG').upper())
    else:
        logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO').upper())

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
        
        # 查找所有媒体文件
        media_files = []
        for pattern in ["*.srt", "*.mp3", "*.mp4"]:
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
        def convert_srt_to_ass(zh_srt_path: Path, en_srt_path: Path, output_dir: Path):
            # 导入 srt2ass.py 中的 fileopen 和 srt2ass 函数 (原始名称)
            from .translation_core.utils.srt2ass import fileopen, srt2ass as srt2ass_original_func

            head_str = '''[Script Info]
; This is an Advanced Sub Station Alpha v4+ script.
Title:
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Serif,18,&H0000FFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,1,1,7,1
Style: Secondary,宋体-简 黑体,11,&H0000FF00,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,1,1,7,1

[Events]
Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text'''

            # 直接调用 srt2ass 函数，并传入文件路径和样式
            # srt2ass.py 的 main 函数中，它会根据文件名中的 'zh' 和 'en' 来决定样式
            # 这里我们直接指定样式
            subLines1 = srt2ass_original_func(str(zh_srt_path), 'Secondary')
            subLines2 = srt2ass_original_func(str(en_srt_path), 'Default')
            
            # 使用 zh_srt_path 来获取编码和基础文件名
            src = fileopen(str(zh_srt_path))
            tmp = src[0]
            encoding = src[1]

            if u'\ufeff' in tmp:
                tmp = tmp.replace(u'\ufeff', '')
            
            tmp = tmp.replace("", "")
            lines = [x.strip() for x in tmp.split("\n") if x.strip()]
            
            # 确保输出文件名是基于原始文件名，而不是带有 .zh 或 .en 的
            base_name = zh_srt_path.stem.replace('.zh', '') # 移除 .zh 后缀
            output_file = output_dir / f"{base_name}.ass"
            
            output_str = head_str + '\n' + subLines1 + subLines2
            output_str = output_str.encode(encoding)

            with open(output_file, 'wb') as output:
                output.write(output_str)
            
            return output_file

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

if __name__ == "__main__":
    app()
