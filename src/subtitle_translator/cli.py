import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

import typer
from typing_extensions import Annotated
import glob
import sys
import time


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
        debug_mode = ('-d' in sys.argv or '--debug' in sys.argv or 
                     os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes'))
        
        from .logger import setup_logger
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
from .translation_core.config import get_default_config, SubtitleConfig
from .translation_core.data import load_subtitle, SubtitleData
from .translation_core.utils.test_openai import test_openai
from .logger import setup_logger, log_section_start, log_section_end, log_stats, create_progress_logger


# 配置日志
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') # 这一行已被移除
# 延迟初始化logger，在setup_environment中初始化
logger = None

class OpenAIAPIError(Exception):
    """OpenAI API 相关错误"""
    pass

class SubtitleTranslatorService:
    def __init__(self):
        self.config = SubtitleConfig()
        self.summarizer = SubtitleSummarizer(self.config)

    def _init_translation_env(self, llm_model: str) -> None:
        """初始化翻译环境并测试连接"""
        start_time = time.time()
        log_section_start(logger, "翻译环境初始化", "⚙️")
        
        if llm_model:
            self.config.split_model = llm_model
            self.config.summary_model = llm_model
            self.config.translation_model = llm_model

        logger.info(f"🌐 API端点: {self.config.openai_base_url}")
        
        model_config = {
            "断句模型": self.config.split_model,
            "总结模型": self.config.summary_model,
            "翻译模型": self.config.translation_model
        }
        log_stats(logger, model_config, "模型配置")
        
        # 使用翻译模型进行连接测试
        logger.info("🔌 正在测试API连接...")
        print("🔌 [bold yellow]测试API连接...[/bold yellow]")
        success, error_msg = test_openai(self.config.openai_base_url, self.config.openai_api_key, self.config.translation_model)
        if not success:
            logger.error(f"❌ API连接测试失败: {error_msg}")
            print(f"[bold red]❌ API连接失败: {error_msg}[/bold red]")
            raise OpenAIAPIError(error_msg)
        
        logger.info("✅ API连接测试成功")
        print("✅ [bold green]API连接成功[/bold green]")
        
        # 显示模型配置
        print(f"🤖 [bold blue]模型配置:[/bold blue]")
        print(f"   断句: [cyan]{self.config.split_model}[/cyan]")
        print(f"   总结: [cyan]{self.config.summary_model}[/cyan]")
        print(f"   翻译: [cyan]{self.config.translation_model}[/cyan]")
        
        elapsed_time = time.time() - start_time
        log_section_end(logger, "翻译环境初始化", elapsed_time, "✅")

    def translate_srt(self, input_srt_path: Path, target_lang: str, output_dir: Path, 
                      llm_model: Optional[str] = None, reflect: bool = False) -> Path:
        """翻译字幕文件"""
        try:
            task_start_time = time.time()
            log_section_start(logger, "字幕翻译任务", "🎬")
            
            # 用于收集各阶段耗时的字典
            stage_times = {}
            
            # 初始化翻译环境
            init_start_time = time.time()
            self._init_translation_env(llm_model)
            stage_times["🔧 环境初始化"] = time.time() - init_start_time
            
            # 加载字幕文件
            logger.info("📂 正在加载字幕文件...")
            asr_data = load_subtitle(str(input_srt_path))
            logger.info(f"📊 字幕统计: 共 {len(asr_data.segments)} 条字幕")
            logger.debug(f"字幕内容预览: {asr_data.to_txt()[:100]}...")  
            
            print(f"📊 [bold blue]加载完成[/bold blue] (共 [cyan]{len(asr_data.segments)}[/cyan] 条字幕)")
            
            # 检查是否需要重新断句
            split_time = 0
            if asr_data.is_word_timestamp():
                section_start_time = time.time()
                log_section_start(logger, "字幕断句处理", "✂️")
                print(f"✂️ [bold yellow]智能断句处理中...[/bold yellow]")
                
                model = self.config.split_model
                logger.info(f"🤖 使用模型: {model}")
                logger.info(f"📏 句子长度限制: {self.config.max_word_count_english} 字")
                
                asr_data = merge_segments(asr_data, model=model, 
                                       num_threads=self.config.thread_num, 
                                       save_split=None)
                
                split_time = time.time() - section_start_time
                log_section_end(logger, "字幕断句处理", split_time, "✅")
                print(f"✅ [bold green]断句完成[/bold green] (优化为 [cyan]{len(asr_data.segments)}[/cyan] 句)")
            
            if split_time > 0:
                stage_times["✂️ 智能断句"] = split_time
            
            # 获取字幕摘要
            summary_start_time = time.time()
            summarize_result = self._get_subtitle_summary(asr_data, str(input_srt_path))
            summary_time = time.time() - summary_start_time
            stage_times["🔍 内容分析"] = summary_time
            
            # 翻译字幕
            translate_start_time = time.time()
            translate_result = self._translate_subtitles(asr_data, summarize_result, reflect)
            translate_time = time.time() - translate_start_time
            mode_name = "🤔 反思翻译" if reflect else "🌍 常规翻译"
            stage_times[mode_name] = translate_time
            
            # 保存字幕
            logger.info("💾 正在保存翻译结果...")
            base_name = input_srt_path.stem
            zh_output_path = output_dir / f"{base_name}.{target_lang}.srt"
            en_output_path = output_dir / f"{base_name}.en.srt"

            asr_data.save_translations_to_files(
                translate_result,
                str(en_output_path),
                str(zh_output_path)
            )
            
            total_elapsed = time.time() - task_start_time
            
            # 显示时间统计
            print()
            self._format_time_stats(stage_times, total_elapsed)
            
            # 任务完成统计
            final_stats = {
                "输入文件": input_srt_path.name,
                "字幕数量": len(asr_data.segments),
                "目标语言": target_lang,
                "翻译模式": "反思翻译" if reflect else "常规翻译",
                "总耗时": f"{total_elapsed:.1f}秒"
            }
            log_stats(logger, final_stats, "任务完成统计")
            log_section_end(logger, "字幕翻译任务", total_elapsed, "🎉")
            
            return zh_output_path
                
        except OpenAIAPIError as e:
            logger.error(f"🚨 API错误: {str(e)}")
            raise typer.Exit(code=1)
            
        except Exception as e:
            logger.error(f"💥 处理过程中发生错误: {str(e)}")
            logger.exception("详细错误信息:")
            raise typer.Exit(code=1)

    def _get_subtitle_summary(self, asr_data: SubtitleData, input_file: str) -> dict:
        """获取字幕内容摘要"""
        section_start_time = time.time()
        log_section_start(logger, "字幕内容分析", "🔍")
        print(f"🔍 [bold cyan]内容分析中...[/bold cyan]")
        
        logger.info(f"🤖 使用模型: {self.config.summary_model}")
        summarize_result = self.summarizer.summarize(asr_data.to_txt(), input_file)
        logger.info(f"总结字幕内容:\n{summarize_result.get('summary')}\n")
        
        section_elapsed = time.time() - section_start_time
        log_section_end(logger, "字幕内容分析", section_elapsed, "✅")
        print(f"✅ [bold green]内容分析完成[/bold green]")
        
        return summarize_result

    def _translate_subtitles(self, asr_data: SubtitleData, summarize_result: dict, reflect: bool = False) -> list:
        """翻译字幕内容"""
        section_start_time = time.time()
        mode_name = "反思翻译" if reflect else "常规翻译"
        log_section_start(logger, f"字幕{mode_name}", "🌍")
        
        print(f"🌍 [bold magenta]{mode_name}中...[/bold magenta] ({len(asr_data.segments)} 句)")
        
        logger.info(f"🤖 使用模型: {self.config.translation_model}")
        logger.info(f"⚡ 线程数: {self.config.thread_num}")
        
        try:
            translator = SubtitleOptimizer(
                config=self.config,
                need_reflect=reflect
            )
            translate_result = translator.translate(asr_data, summarize_result)
            
            # 获取优化统计
            stats = self._get_optimization_stats(translator.batch_logs, reflect)
            
            section_elapsed = time.time() - section_start_time
            log_section_end(logger, f"字幕{mode_name}", section_elapsed, "🎉")
            print(f"✅ [bold green]{mode_name}完成[/bold green]")
            
            # 显示优化统计
            if stats['total_changes'] > 0:
                print(f"📊 [bold blue]优化统计:[/bold blue]")
                if stats['format_changes'] > 0:
                    print(f"   格式优化: [cyan]{stats['format_changes']}[/cyan] 项")
                if stats['content_changes'] > 0:
                    print(f"   内容修改: [cyan]{stats['content_changes']}[/cyan] 项")
                if stats['reflect_changes'] > 0:
                    print(f"   反思优化: [cyan]{stats['reflect_changes']}[/cyan] 项")
                if stats['wrong_changes'] > 0:
                    print(f"   [yellow]可疑替换: {stats['wrong_changes']} 项[/yellow]")
                print(f"   总计: [cyan]{stats['total_changes']}[/cyan] 项优化")
            else:
                print("📊 [dim]无需优化调整[/dim]")
            
            return translate_result
        except Exception as e:
            logger.error(f"❌ 翻译失败: {str(e)}")
            print(f"[bold red]❌ 翻译失败: {str(e)}[/bold red]")
            raise

    def _get_optimization_stats(self, batch_logs: list, reflect: bool) -> dict:
        """从batch_logs中获取优化统计信息"""
        import string
        
        def is_format_change_only(original, optimized):
            """判断是否只有格式变化（大小写和标点符号）"""
            # 忽略大小写和标点符号后比较
            original_normalized = original.lower().translate(str.maketrans('', '', string.punctuation))
            optimized_normalized = optimized.lower().translate(str.maketrans('', '', string.punctuation))
            return original_normalized == optimized_normalized

        def is_wrong_replacement(original, optimized):
            """检测是否存在错误的替换（替换了不相关的词）"""
            import re
            # 提取所有单词
            original_words = set(re.findall(r'\b\w+\b', original.lower()))
            optimized_words = set(re.findall(r'\b\w+\b', optimized.lower()))
            # 找出被替换的词
            removed_words = original_words - optimized_words
            added_words = optimized_words - original_words
            # 如果替换前后的词没有相似性，可能是错误替换
            if removed_words and added_words:
                for removed in removed_words:
                    for added in added_words:
                        # 如果原词和新词完全不同（编辑距离过大），判定为错误替换
                        if len(removed) > 3 and len(added) > 3 and not any(c in removed for c in added):
                            return True
            return False

        # 统计变更类型
        format_changes = 0
        content_changes = 0
        wrong_changes = 0
        reflect_changes = 0

        # 遍历所有日志
        for log in batch_logs:
            if log["type"] == "content_optimization":
                original = log["original"]
                optimized = log["optimized"]
                
                # 分类统计
                if is_format_change_only(original, optimized):
                    format_changes += 1
                elif is_wrong_replacement(original, optimized):
                    wrong_changes += 1
                else:
                    content_changes += 1
            
            elif log["type"] == "reflect_translation":
                reflect_changes += 1

        total_changes = format_changes + content_changes + wrong_changes + reflect_changes
        
        return {
            'format_changes': format_changes,
            'content_changes': content_changes,
            'wrong_changes': wrong_changes,
            'reflect_changes': reflect_changes,
            'total_changes': total_changes
        }

    def _format_time_stats(self, stages: dict, total_time: float) -> None:
        """格式化显示时间统计"""
        print(f"⏱️  [bold blue]耗时统计:[/bold blue]")
        
        # 按时间排序显示各阶段
        sorted_stages = sorted(stages.items(), key=lambda x: x[1], reverse=True)
        
        for stage_name, elapsed_time in sorted_stages:
            if elapsed_time > 0:
                percentage = (elapsed_time / total_time) * 100
                print(f"   {stage_name}: [cyan]{elapsed_time:.1f}s[/cyan] ([dim]{percentage:.0f}%[/dim])")
        
        print(f"   [bold]总计: [cyan]{total_time:.1f}s[/cyan][/bold]")

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
        logger.info(f"开始处理单个文件: {input_file.name}")
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
                print(f"📄 发现文件 [cyan]{input_file_found}[/cyan]")
            else:
                print(f"❌ 没有找到 [yellow]{base_name}[/yellow] 的输入文件")
        
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
        print()
        logger.info(f"🎯 处理文件 ({i+1}/{len(files_to_process)}): {current_input_file.name}")
        print(f"🎯 处理文件 ({i+1}/{len(files_to_process)}): [bold cyan]{current_input_file.name}[/bold cyan]")
        
        try:
            _process_single_file(
                current_input_file, target_lang, output_dir, model, 
                llm_model, reflect, debug
            )
            count += 1
            logger.info(f"✅ {current_input_file.stem} 处理完成！")
            print(f"[bold green]✅ {current_input_file.stem} 处理完成！[/bold green]")
            
            # 检查是否生成了ASS文件
            ass_file = output_dir / f"{current_input_file.stem}.ass"
            if ass_file.exists():
                logger.info(f"📺 双语ASS文件已生成: {ass_file.name}")
                print(f"📺 双语ASS文件已生成: [cyan]{ass_file.name}[/cyan]")
        
        except Exception as e:
            print(f"[bold red]❌ {current_input_file.stem} 处理失败！{e}[/bold red]")
        
        print()  # 添加空行分隔
    
    # 显示处理结果 - 简化输出
    print()
    logger.info("🎉 批量处理完成！")
    logger.info(f"总计处理文件数: {count}")
    print(f"🎉 [bold green]批量处理完成！[/bold green] (处理 [cyan]{count}[/cyan] 个文件)")
    
    # 只显示生成的ASS文件统计，不显示详细列表
    if count > 0:
        ass_files = list(output_dir.glob("*.ass"))
        if ass_files:
            logger.info("生成的文件：")
            for f in ass_files:
                logger.info(f"  {f.name}")
            print(f"📺 [bold green]已生成 {len(ass_files)} 个双语ASS文件[/bold green]")
        
        srt_files = [f for f in output_dir.glob("*.srt") if not ("_zh" in f.name or "_en" in f.name)]
        if srt_files:
            logger.info("原始字幕文件：")
            for f in srt_files:
                logger.info(f"  {f.name}")
    
    logger.info("处理完毕！")


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
        logger.info(">>> 开始转录...")
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
            
            # 统计字幕数量
            subtitle_count = len(srt_content.strip().split('\n\n'))
            logger.info(f"转录完成，SRT文件保存至: {temp_srt_path}")
            print(f"✅ [bold green]转录完成[/bold green] (共 [cyan]{subtitle_count}[/cyan] 条字幕)")

        except Exception as e:
            print(f"[bold red]转录失败:[/bold red] {e}")
            raise typer.Exit(code=1)

    final_translated_zh_path = None
    final_translated_en_path = None

    # --- 翻译阶段 ---
    logger.info(">>> 开始翻译...")
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

        logger.info(f"翻译完成，中文翻译文件保存至: {final_translated_zh_path}")
        logger.info(f"英文翻译文件保存至: {final_translated_en_path}")

        # --- 转换为 ASS ---
        print(">>> [bold green]生成双语ASS文件...[/bold green]")
        logger.info(">>> 正在转换为 ASS 格式...")

        # 提取 srt2ass.py 的核心逻辑
        from .translation_core.utils.ass_converter import convert_srt_to_ass

        final_ass_path = convert_srt_to_ass(final_translated_zh_path, final_translated_en_path, output_dir)
        logger.info(f"ASS 文件生成成功: {final_ass_path}")

    except Exception as e:
        print(f"[bold red]翻译或 ASS 转换失败:[/bold red] {e}")
        raise typer.Exit(code=1)
    finally:
        # --- 清理中间翻译文件，保留原始转录文件 ---
        logger.info(">>> 正在清理中间翻译文件...")
        cleaned_files = 0
        if final_translated_zh_path and final_translated_zh_path.exists():
            os.remove(final_translated_zh_path)
            logger.info(f"已删除中间文件: {final_translated_zh_path}")
            cleaned_files += 1
        if final_translated_en_path and final_translated_en_path.exists():
            os.remove(final_translated_en_path)
            logger.info(f"已删除中间文件: {final_translated_en_path}")
            cleaned_files += 1
        
        if cleaned_files > 0:
            print(f"🧹 已清理 {cleaned_files} 个中间文件")
        
        # 处理原始SRT文件
        if temp_srt_path and temp_srt_path.exists():
            if input_file.suffix.lower() != '.srt':
                logger.info(f"保留原始转录文件: {temp_srt_path}")
                print(f"💾 [bold green]保留转录文件:[/bold green] [cyan]{temp_srt_path.name}[/cyan]")

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
