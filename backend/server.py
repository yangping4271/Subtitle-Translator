#!/usr/bin/env python3
"""
YouTube 双语字幕翻译后端服务
支持音频缓存、字幕缓存和翻译缓存
"""

import os
import sys
import uuid
import json
import asyncio
import subprocess
import traceback
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 设置环境变量加载
def setup_project_environment():
    """设置项目环境变量"""
    try:
        # 添加项目源码路径到Python路径
        project_root = Path(__file__).parent.parent
        src_path = project_root / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        # 加载项目环境配置
        from subtitle_translator.env_setup import setup_environment
        setup_environment()
        print("✅ 项目环境配置已加载")
        
    except Exception as e:
        print(f"⚠️  环境配置加载失败: {e}")
        print("使用默认环境变量")

# 初始化环境
setup_project_environment()

# 全局状态
JOBS: Dict[str, Dict[str, Any]] = {}
TMP_DIR = os.path.join("/tmp", "yt_subs")
CACHE_DIR = os.path.join("/tmp", "yt_cache")
AUDIO_CACHE_DIR = os.path.join(CACHE_DIR, "audio")
SUBTITLE_CACHE_DIR = os.path.join(CACHE_DIR, "subtitles")
TRANSLATION_CACHE_DIR = os.path.join(CACHE_DIR, "translations")

# 创建必要的目录
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
os.makedirs(SUBTITLE_CACHE_DIR, exist_ok=True)
os.makedirs(TRANSLATION_CACHE_DIR, exist_ok=True)

# 缓存元数据
CACHE_METADATA: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="YouTube 双语字幕翻译服务")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscribeRequest(BaseModel):
    url: str
    target_lang: str = "zh"

# 工具函数
def _extract_video_id(url: str) -> Optional[str]:
    """从 YouTube URL 提取视频 ID"""
    if "youtube.com/watch" in url or "youtu.be/" in url:
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
    return None

def _get_cache_key(video_id: str, cache_type: str) -> str:
    """生成缓存键"""
    return f"{video_id}_{cache_type}"

def _get_cached_audio_path(video_id: str) -> Optional[str]:
    """获取缓存的音频文件路径"""
    cache_key = _get_cache_key(video_id, "audio")
    if cache_key in CACHE_METADATA:
        cache_info = CACHE_METADATA[cache_key]
        audio_path = cache_info.get("path")
        if audio_path and os.path.exists(audio_path):
            cache_info["last_accessed"] = datetime.utcnow().isoformat()
            return audio_path
    return None

def _get_cached_subtitle_path(video_id: str) -> Optional[str]:
    """获取缓存的字幕文件路径"""
    cache_key = _get_cache_key(video_id, "subtitle")
    if cache_key in CACHE_METADATA:
        cache_info = CACHE_METADATA[cache_key]
        subtitle_path = cache_info.get("path")
        if subtitle_path and os.path.exists(subtitle_path):
            cache_info["last_accessed"] = datetime.utcnow().isoformat()
            return subtitle_path
    return None

def _get_cached_translation_files(video_id: str, target_lang: str = "zh") -> Dict[str, Optional[str]]:
    """获取缓存的翻译文件路径"""
    cache_key = _get_cache_key(video_id, f"translation_{target_lang}")
    result = {
        "english_srt": None,
        "translated_srt": None,
        "ass_file": None
    }
    
    if cache_key in CACHE_METADATA:
        cache_info = CACHE_METADATA[cache_key]
        cache_dir = cache_info.get("cache_dir")
        
        if cache_dir and os.path.exists(cache_dir):
            # 查找文件
            for file in os.listdir(cache_dir):
                if file.endswith('.en.srt'):
                    result["english_srt"] = os.path.join(cache_dir, file)
                elif file.endswith(f'.{target_lang}.srt'):
                    result["translated_srt"] = os.path.join(cache_dir, file)
                elif file.endswith('.ass'):
                    result["ass_file"] = os.path.join(cache_dir, file)
            
            # 验证所有文件都存在
            if all(path and os.path.exists(path) for path in result.values()):
                cache_info["last_accessed"] = datetime.utcnow().isoformat()
                return result
    
    return {"english_srt": None, "translated_srt": None, "ass_file": None}

def _cache_translation_files(video_id: str, temp_dir: str, target_lang: str = "zh") -> bool:
    """缓存翻译文件"""
    try:
        cache_key = _get_cache_key(video_id, f"translation_{target_lang}")
        cache_dir = os.path.join(TRANSLATION_CACHE_DIR, video_id)
        os.makedirs(cache_dir, exist_ok=True)
        
        # 复制所有生成的文件到缓存目录
        files_copied = 0
        total_size = 0
        
        for file in os.listdir(temp_dir):
            if file.endswith(('.srt', '.ass')):
                src_path = os.path.join(temp_dir, file)
                dst_path = os.path.join(cache_dir, file)
                shutil.copy2(src_path, dst_path)
                files_copied += 1
                total_size += os.path.getsize(dst_path)
        
        if files_copied > 0:
            CACHE_METADATA[cache_key] = {
                "video_id": video_id,
                "cache_dir": cache_dir,
                "target_lang": target_lang,
                "files_count": files_copied,
                "size": total_size,
                "created_at": datetime.utcnow().isoformat(),
                "last_accessed": datetime.utcnow().isoformat()
            }
            print(f"✅ 翻译缓存已保存: {video_id} ({files_copied} 文件, {total_size} 字节)")
            return True
        
    except Exception as e:
        print(f"❌ 翻译缓存失败: {e}")
        
    return False

def _extract_subtitle_text(subtitle_path: str) -> str:
    """从字幕文件提取文本内容，用于优化总结和翻译"""
    try:
        if not subtitle_path or not os.path.exists(subtitle_path):
            print(f"字幕文件不存在: {subtitle_path}")
            return ""
            
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 简单提取文本（跳过时间戳和序号）
        lines = content.split('\n')
        text_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.isdigit() and '-->' not in line:
                text_lines.append(line)
        
        return ' '.join(text_lines)
    except Exception as e:
        print(f"提取字幕文本失败: {e}")
        return ""

def _parse_srt_to_segments(srt_path: str) -> List[Dict[str, Any]]:
    """解析 SRT 文件为 segments 格式"""
    segments = []
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        blocks = content.strip().split('\n\n')
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # 解析时间
                time_line = lines[1]
                if '-->' in time_line:
                    start_time, end_time = time_line.split(' --> ')
                    text = '\n'.join(lines[2:])
                    
                    segments.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": text.strip()
                    })
    except Exception as e:
        print(f"解析 SRT 文件失败: {e}")
    
    return segments

# API 端点
@app.get("/config")
async def get_config():
    """获取后台配置信息（不暴露敏感信息）"""
    import os
    
    config_info = {
        "status": "ok",
        "models": {
            "split_model": os.getenv('SPLIT_MODEL', 'gpt-4o-mini'),
            "translation_model": os.getenv('TRANSLATION_MODEL', 'gpt-4o'),
            "summary_model": os.getenv('SUMMARY_MODEL', 'gpt-4o-mini'),
            "llm_model": os.getenv('LLM_MODEL', 'gpt-4o-mini')
        },
        "api_configured": bool(os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_BASE_URL')),
        "hf_endpoint": os.getenv('HF_ENDPOINT', ''),
        "available_languages": ["zh", "zh-tw", "ja", "ko", "en", "fr", "de", "es"],
        "backend_version": "0.3.0"
    }
    
    return config_info

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "jobs_count": len(JOBS),
        "cache_entries": len(CACHE_METADATA)
    }

@app.post("/translate_youtube")
async def translate_youtube(req: TranscribeRequest, bg: BackgroundTasks):
    """使用项目完整的translate命令处理YouTube视频"""
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "progress": {},
        "events": [],
        "error": None,
        "youtube_url": req.url,
        "target_lang": req.target_lang
    }
    
    bg.add_task(_run_translate_job, job_id, req.url, req.target_lang)
    return {"job_id": job_id}

async def _run_translate_job(job_id: str, url: str, target_lang: str = "zh"):
    """执行翻译作业"""
    job = JOBS[job_id]
    job["status"] = "running"
    job["progress"] = {"stage": "initializing", "message": "开始处理..."}

    try:
        vid = _extract_video_id(url)
        if not vid:
            raise RuntimeError("invalid youtube url")

        # 检查翻译缓存
        cached_translation = _get_cached_translation_files(vid, target_lang)
        if all(cached_translation.values()):
            print(f"🚀 翻译缓存命中: {vid}")
            job["progress"].update({
                "stage": "cache_hit", 
                "message": "翻译缓存命中，直接加载完成"
            })
            
            # 从缓存的 SRT 文件生成 events
            english_segments = _parse_srt_to_segments(cached_translation["english_srt"])
            translated_segments = _parse_srt_to_segments(cached_translation["translated_srt"])
            
            # 合并双语 segments
            events = []
            for i, (en_seg, zh_seg) in enumerate(zip(english_segments, translated_segments)):
                events.append({
                    "id": str(i),
                    "start_time": en_seg["start_time"],
                    "end_time": en_seg["end_time"],
                    "text": en_seg["text"],
                    "translation": zh_seg["text"]
                })
            
            job["events"] = events
            job["translation_cache"] = cached_translation
            job["progress"].update({
                "stage": "completed", 
                "message": f"翻译缓存加载完成，生成{len(events)}个字幕段"
            })
            job["status"] = "done"
            return

        job["progress"].update({"stage": "downloading", "message": "开始下载音频..."})

        # 检查音频缓存
        cached_audio_path = _get_cached_audio_path(vid)
        cached_subtitle_path = _get_cached_subtitle_path(vid)

        with tempfile.TemporaryDirectory() as temp_dir:
            actual_audio_file = None
            
            if cached_audio_path:
                print(f"🚀 音频缓存命中: {vid}")
                job["progress"].update({"stage": "cache_hit", "message": "音频缓存命中，跳过下载"})
                actual_audio_file = cached_audio_path
            else:
                # 下载音频
                job["progress"].update({"stage": "downloading_audio", "message": "下载音频文件..."})
                
                audio_file = os.path.join(temp_dir, f"{vid}.mp3")
                download_cmd = [
                    "yt-dlp", 
                    "--extract-audio", 
                    "--audio-format", "mp3",
                    "--output", audio_file,
                    url
                ]
                
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *download_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
                    
                    if proc.returncode != 0:
                        raise RuntimeError(f"音频下载失败: {stderr.decode()}")
                    
                    actual_audio_file = audio_file
                    
                    # 缓存音频文件
                    cached_path = os.path.join(AUDIO_CACHE_DIR, f"{vid}.mp3")
                    shutil.copy2(actual_audio_file, cached_path)
                    
                    cache_key = _get_cache_key(vid, "audio")
                    CACHE_METADATA[cache_key] = {
                        "video_id": vid,
                        "path": cached_path,
                        "size": os.path.getsize(cached_path),
                        "created_at": datetime.utcnow().isoformat(),
                        "last_accessed": datetime.utcnow().isoformat()
                    }
                    
                except asyncio.TimeoutError:
                    raise RuntimeError("音频下载超时")

            # 下载字幕（如果没有缓存）
            if not cached_subtitle_path:
                job["progress"].update({"stage": "downloading_subtitle", "message": "尝试下载字幕..."})
                
                subtitle_file = os.path.join(temp_dir, f"{vid}.vtt")
                subtitle_cmd = [
                    "yt-dlp",
                    "--write-auto-sub",
                    "--sub-lang", "en",
                    "--sub-format", "vtt",
                    "--skip-download",
                    "--output", os.path.join(temp_dir, f"{vid}.%(ext)s"),
                    url
                ]
                
                try:
                    proc_sub = await asyncio.create_subprocess_exec(
                        *subtitle_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout_sub, stderr_sub = await asyncio.wait_for(proc_sub.communicate(), timeout=30)
                    
                    # 查找生成的字幕文件
                    vtt_files = [f for f in os.listdir(temp_dir) if f.endswith('.vtt')]
                    if vtt_files and os.path.exists(os.path.join(temp_dir, vtt_files[0])):
                        subtitle_file = os.path.join(temp_dir, vtt_files[0])
                        
                        # 缓存字幕文件
                        cached_subtitle_path = os.path.join(SUBTITLE_CACHE_DIR, f"{vid}.vtt")
                        shutil.copy2(subtitle_file, cached_subtitle_path)
                        
                        cache_key = _get_cache_key(vid, "subtitle")
                        CACHE_METADATA[cache_key] = {
                            "video_id": vid,
                            "path": cached_subtitle_path,
                            "size": os.path.getsize(cached_subtitle_path),
                            "created_at": datetime.utcnow().isoformat(),
                            "last_accessed": datetime.utcnow().isoformat()
                        }
                        
                        job["progress"].update({"subtitle_context_file": cached_subtitle_path})
                        
                except asyncio.TimeoutError:
                    print("字幕下载超时，继续使用音频转录")
                except Exception as e:
                    print(f"字幕下载失败: {e}")

            # 准备翻译命令
            job["progress"].update({"stage": "transcribing", "message": "开始转录和翻译..."})
            
            # 使用项目的translate命令处理，带有preserve-intermediate选项
            translate_cmd = [
                sys.executable, "-m", "subtitle_translator.cli",
                "-i", actual_audio_file,
                "-t", target_lang,
                "-o", temp_dir,
                "-r",  # 启用反思模式
                "--preserve-intermediate"  # 保留中间SRT文件
            ]

            # 准备原始字幕文本用于优化总结
            subtitle_context = ""
            subtitle_file = cached_subtitle_path
            if subtitle_file and os.path.exists(subtitle_file):
                subtitle_context = _extract_subtitle_text(subtitle_file)
                if subtitle_context:
                    job["progress"].update({"subtitle_context_file": subtitle_file})

            print(f"执行翻译命令: {' '.join(translate_cmd)}")
            
            proc = await asyncio.create_subprocess_exec(
                *translate_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                error_msg = (stderr or b'').decode('utf-8', 'ignore')
                
                # 提供更详细的错误信息
                if "API调用失败" in error_msg or "API Key" in error_msg:
                    raise RuntimeError(f"API配置错误: {error_msg}")
                elif "模型下载" in error_msg or "网络" in error_msg:
                    raise RuntimeError(f"模型下载失败: {error_msg}")
                elif "文件不存在" in error_msg:
                    raise RuntimeError(f"输入文件问题: {error_msg}")
                else:
                    raise RuntimeError(f"translate命令失败 (退出码 {proc.returncode}): {error_msg[-500:]}")

            # 查找生成的文件
            srt_files = [f for f in os.listdir(temp_dir) if f.endswith('.srt')]
            ass_files = [f for f in os.listdir(temp_dir) if f.endswith('.ass')]
            
            # 查找英文和翻译的SRT文件
            english_srt = None
            translated_srt = None
            
            for file in srt_files:
                if file.endswith('.en.srt'):
                    english_srt = os.path.join(temp_dir, file)
                elif file.endswith(f'.{target_lang}.srt'):
                    translated_srt = os.path.join(temp_dir, file)
            
            if english_srt and translated_srt and os.path.exists(english_srt) and os.path.exists(translated_srt):
                # 解析SRT文件生成events
                english_segments = _parse_srt_to_segments(english_srt)
                translated_segments = _parse_srt_to_segments(translated_srt)
                
                # 合并双语segments
                events = []
                for i, (en_seg, zh_seg) in enumerate(zip(english_segments, translated_segments)):
                    events.append({
                        "id": str(i),
                        "start_time": en_seg["start_time"],
                        "end_time": en_seg["end_time"],
                        "text": en_seg["text"],
                        "translation": zh_seg["text"]
                    })
                
                job["events"] = events
                
                # 缓存翻译文件
                _cache_translation_files(vid, temp_dir, target_lang)
                
                if ass_files:
                    job["translation_file"] = ass_files[0]
                    
                job["progress"].update({
                    "stage": "completed", 
                    "message": f"翻译完成，生成{len(events)}个字幕段"
                })
            else:
                raise RuntimeError("翻译文件生成失败：未找到预期的SRT文件")

        job["status"] = "done"

    except Exception as e:
        print(f"作业 {job_id} 失败: {e}")
        job["status"] = "error"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()

@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    """获取作业状态"""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", {}),
        "error": job.get("error"),
        "events_count": len(job.get("events", []))
    }

@app.get("/video/{video_id}/segments")
async def get_video_segments(video_id: str, target_lang: str = "zh"):
    """根据视频ID获取翻译segments（用于扩展重新加载缓存）"""
    try:
        # 检查是否有翻译缓存
        cached_translation = _get_cached_translation_files(video_id, target_lang)
        if all(cached_translation.values()):
            # 从缓存的 SRT 文件生成 segments
            english_segments = _parse_srt_to_segments(cached_translation["english_srt"])
            translated_segments = _parse_srt_to_segments(cached_translation["translated_srt"])
            
            # 合并双语 segments
            events = []
            for i, (en_seg, zh_seg) in enumerate(zip(english_segments, translated_segments)):
                events.append({
                    "id": str(i),
                    "start_time": en_seg["start_time"],
                    "end_time": en_seg["end_time"],
                    "text": en_seg["text"],
                    "translation": zh_seg["text"]
                })
            
            return {
                "video_id": video_id,
                "target_lang": target_lang,
                "segments": events,
                "total_count": len(events),
                "source": "translation_cache"
            }
        else:
            # 检查是否有正在进行的任务
            active_jobs = []
            for job_id, job_data in JOBS.items():
                if (job_data.get("youtube_url", "").find(f"v={video_id}") != -1 and 
                    job_data.get("target_lang") == target_lang):
                    active_jobs.append({
                        "job_id": job_id,
                        "status": job_data.get("status"),
                        "events_count": len(job_data.get("events", []))
                    })
            
            return {
                "video_id": video_id,
                "target_lang": target_lang,
                "segments": [],
                "total_count": 0,
                "source": "no_cache",
                "active_jobs": active_jobs
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取视频segments失败: {str(e)}")

@app.get("/segments/{job_id}")
async def get_segments(job_id: str, start: float = 0, window: int = 60):
    """获取指定作业的字幕段落"""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.get("status") != "done":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    events = job.get("events", [])
    
    # 简单的时间窗口过滤（这里简化处理）
    window_events = events[int(start):int(start + window)] if start < len(events) else []
    
    return {
        "job_id": job_id,
        "segments": window_events,
        "total_count": len(events),
        "window_start": start,
        "window_size": len(window_events)
    }

@app.get("/srt_files/{job_id}")
async def get_srt_files(job_id: str):
    """获取作业的SRT文件（双语字幕）"""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") != "done":
        raise HTTPException(status_code=400, detail="Job not completed")

    # 优先从翻译缓存获取
    vid = _extract_video_id(job.get("youtube_url", ""))
    target_lang = job.get("target_lang", "zh")
    
    if vid:
        cached_translation = _get_cached_translation_files(vid, target_lang)
        if all(cached_translation.values()):
            try:
                # 读取SRT文件内容
                with open(cached_translation["english_srt"], 'r', encoding='utf-8') as f:
                    english_srt_content = f.read()
                
                with open(cached_translation["translated_srt"], 'r', encoding='utf-8') as f:
                    translated_srt_content = f.read()
                
                return {
                    "job_id": job_id,
                    "video_id": vid,
                    "target_lang": target_lang,
                    "has_english_srt": True,
                    "has_translated_srt": True,
                    "english_srt": english_srt_content,
                    "translated_srt": translated_srt_content,
                    "english_preview": english_srt_content[:200] + "...",
                    "translated_preview": translated_srt_content[:200] + "...",
                    "source": "translation_cache"
                }
            except Exception as e:
                print(f"读取缓存SRT文件失败: {e}")

    # 如果缓存不可用，返回空结果
    return {
        "job_id": job_id,
        "video_id": vid,
        "target_lang": target_lang,
        "has_english_srt": False,
        "has_translated_srt": False,
        "english_srt": "",
        "translated_srt": "",
        "source": "not_available"
    }

@app.get("/cache/status")
async def cache_status():
    """获取缓存状态"""
    audio_caches = {k: v for k, v in CACHE_METADATA.items() if k.endswith("_audio")}
    subtitle_caches = {k: v for k, v in CACHE_METADATA.items() if k.endswith("_subtitle")}
    translation_caches = {k: v for k, v in CACHE_METADATA.items() if "translation" in k}

    total_audio_size = sum(cache.get("size", 0) for cache in audio_caches.values())
    total_subtitle_size = sum(cache.get("size", 0) for cache in subtitle_caches.values())
    total_translation_size = sum(cache.get("size", 0) for cache in translation_caches.values())

    return {
        "audio_cache": {
            "count": len(audio_caches),
            "total_size": total_audio_size,
            "entries": list(audio_caches.values())
        },
        "subtitle_cache": {
            "count": len(subtitle_caches),
            "total_size": total_subtitle_size,
            "entries": list(subtitle_caches.values())
        },
        "translation_cache": {
            "count": len(translation_caches),
            "total_size": total_translation_size,
            "entries": list(translation_caches.values())
        },
        "total_cache_size": total_audio_size + total_subtitle_size + total_translation_size
    }

@app.get("/cache/check/{video_id}")
async def check_cache(video_id: str):
    """检查指定视频的缓存状态"""
    audio_cache = _get_cached_audio_path(video_id)
    subtitle_cache = _get_cached_subtitle_path(video_id)
    translation_cache = _get_cached_translation_files(video_id)

    result = {
        "video_id": video_id,
        "has_audio_cache": audio_cache is not None,
        "has_subtitle_cache": subtitle_cache is not None,
        "has_translation_cache": all(translation_cache.values()),
        "audio_cache_path": audio_cache,
        "subtitle_cache_path": subtitle_cache,
        "translation_cache_files": translation_cache
    }

    if audio_cache:
        result["audio_cache_size"] = os.path.getsize(audio_cache)
    
    if subtitle_cache:
        result["subtitle_cache_size"] = os.path.getsize(subtitle_cache)

    return result

@app.delete("/cache/clear/{video_id}")
async def clear_cache(video_id: str):
    """清除指定视频的缓存"""
    removed = []

    # 清除音频缓存
    audio_path = _get_cached_audio_path(video_id)
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)
        audio_key = _get_cache_key(video_id, "audio")
        if audio_key in CACHE_METADATA:
            del CACHE_METADATA[audio_key]
        removed.append(f"audio: {audio_path}")

    # 清除字幕缓存
    subtitle_path = _get_cached_subtitle_path(video_id)
    if subtitle_path and os.path.exists(subtitle_path):
        os.remove(subtitle_path)
        subtitle_key = _get_cache_key(video_id, "subtitle")
        if subtitle_key in CACHE_METADATA:
            del CACHE_METADATA[subtitle_key]
        removed.append(f"subtitle: {subtitle_path}")

    # 清除翻译缓存
    translation_cache_dir = os.path.join(TRANSLATION_CACHE_DIR, video_id)
    if os.path.exists(translation_cache_dir):
        shutil.rmtree(translation_cache_dir)
        # 清除相关的元数据
        keys_to_remove = [k for k in CACHE_METADATA.keys() if k.startswith(f"{video_id}_translation")]
        for key in keys_to_remove:
            del CACHE_METADATA[key]
        removed.append(f"translation_dir: {translation_cache_dir}")

    return {
        "video_id": video_id,
        "removed": removed,
        "count": len(removed)
    }

@app.get("/test/download")
async def test_download():
    """测试下载功能"""
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll
    
    try:
        # 测试 yt-dlp 是否可用
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode == 0:
            version = stdout.decode().strip()
            return {
                "status": "ok",
                "yt_dlp_version": version,
                "message": "下载工具可用"
            }
        else:
            return {
                "status": "error",
                "message": "yt-dlp 不可用",
                "error": stderr.decode()
            }
    except Exception as e:
        return {
            "status": "error",
            "message": "测试下载功能失败",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    print("🚀 启动 YouTube 双语字幕翻译服务...")
    print("📡 服务地址: http://127.0.0.1:9009")
    print("📚 API 文档: http://127.0.0.1:9009/docs")
    print("💾 缓存目录:")
    print(f"   - 音频缓存: {AUDIO_CACHE_DIR}")
    print(f"   - 字幕缓存: {SUBTITLE_CACHE_DIR}")
    print(f"   - 翻译缓存: {TRANSLATION_CACHE_DIR}")
    print("=" * 50)
    
    uvicorn.run(app, host="127.0.0.1", port=9009)