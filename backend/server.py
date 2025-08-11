import os
import re
import uuid
import json
import shutil
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: Dict[str, Dict[str, Any]] = {}
TMP_DIR = os.path.join("/tmp", "yt_subs")
os.makedirs(TMP_DIR, exist_ok=True)

class TranscribeRequest(BaseModel):
    youtube_url: str
    chunk_ms: Optional[int] = 20000
    overlap_ms: Optional[int] = 500

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/transcribe")
async def transcribe(req: TranscribeRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "progress": {},
        "events": [],
        "error": None,
    }
    bg.add_task(_run_job, job_id, req.youtube_url)
    return {"job_id": job_id}

@app.get("/jobs/{job_id}/state")
async def job_state(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "status": job["status"],
        "progress": job.get("progress", {}),
        "error": job.get("error"),
    }

@app.get("/segments")
async def segments(
    job_id: str = Query(...),
    from_ms: int = Query(0, ge=0),
    to_ms: int = Query(0, ge=0),
    include_translation: bool = Query(False)
):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    events = job.get("events", [])
    if to_ms <= 0:
        window = events
    else:
        window = [e for e in events if _overlap(e.get("tStartMs", 0), e.get("dDurationMs", 0), from_ms, to_ms)]
    if include_translation:
        # 预留：返回 translation 字段（当前为空）
        pass
    return {"events": window}


def _overlap(start_ms: int, dur_ms: int, from_ms: int, to_ms: int) -> bool:
    a1 = start_ms
    a2 = start_ms + max(0, dur_ms)
    b1 = from_ms
    b2 = to_ms
    return max(a1, b1) < min(a2, b2)

async def _run_job(job_id: str, url: str):
    job = JOBS[job_id]
    job["status"] = "running"
    try:
        vid = _extract_video_id(url)
        if not vid:
            raise RuntimeError("invalid youtube url")
        out_prefix = os.path.join(TMP_DIR, f"yt_{vid}")

        # 1) 使用 yt-dlp 下载英文自动字幕（vtt）
        cmd = [
            "yt-dlp", "--skip-download",
            "--write-auto-subs", "--sub-lang", "en",
            "--sub-format", "vtt",
            "-o", out_prefix,
            url,
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await proc.communicate()

        # 可能的字幕文件名
        vtt_path = f"{out_prefix}.en.vtt"
        if not os.path.exists(vtt_path):
            # 多数模板会生成 .vtt 或 .en.vtt，尝试通配
            alt = [p for p in os.listdir(TMP_DIR) if p.startswith(f"yt_{vid}") and p.endswith(".vtt")]
            if alt:
                vtt_path = os.path.join(TMP_DIR, alt[0])
        events: List[Dict[str, Any]] = []
        if os.path.exists(vtt_path):
            with open(vtt_path, "r", encoding="utf-8", errors="ignore") as f:
                vtt_text = f.read()
            events = _parse_vtt_to_events(vtt_text)

        # 若 vtt 为空，后续可接入调用本项目 uv 工具进行转录（TODO）
        job["events"] = events
        job["status"] = "done"

        # 清理
        try:
            for p in os.listdir(TMP_DIR):
                if p.startswith(f"yt_{vid}"):
                    os.remove(os.path.join(TMP_DIR, p))
        except Exception:
            pass

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


def _extract_video_id(url: str) -> Optional[str]:
    m = re.search(r"(?:v=|/shorts/|youtu\.be/)([A-Za-z0-9_-]{6,})", url)
    return m.group(1) if m else None

_time_re = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})")

def _parse_vtt_to_events(vtt: str) -> List[Dict[str, Any]]:
    lines = vtt.splitlines()
    i = 0
    events: List[Dict[str, Any]] = []
    # 跳过头
    while i < len(lines) and "-->" not in lines[i]:
        i += 1
    while i < len(lines):
        m = _time_re.search(lines[i])
        if m:
            s_ms = (((int(m.group(1))*60 + int(m.group(2)))*60 + int(m.group(3)))*1000 + int(m.group(4)))
            e_ms = (((int(m.group(5))*60 + int(m.group(6)))*60 + int(m.group(7)))*1000 + int(m.group(8)))
            i += 1
            text = []
            while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                text.append(lines[i].strip())
                i += 1
            full = " ".join(text).strip()
            if full:
                events.append({
                    "tStartMs": s_ms,
                    "dDurationMs": max(0, e_ms - s_ms),
                    "segs": [{"utf8": full}],
                })
        else:
            i += 1
    return events

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9009)
