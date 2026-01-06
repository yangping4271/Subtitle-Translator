"""
YouTube SubtitlePlus - Local Subtitle Server
Automatically serves local ASS subtitle files for YouTube videos.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading

# Import core functionality
from ..processor import process_single_file
from ..service import SubtitleTranslatorService
from ..cli import DEFAULT_TRANSCRIPTION_MODEL
from ..env_setup import setup_environment

# Configure logger
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "subtitle_dirs": ["~/Downloads", "~/subtitles"],
    "server_port": 8888,
    "server_host": "127.0.0.1",
    "supported_formats": [".ass", ".srt", ".vtt"],
    "cors_origins": ["chrome-extension://*", "http://localhost:*"]
}

class SubtitleServer:
    def __init__(self, config: Optional[dict] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        self.subtitle_dirs = self.setup_subtitle_dirs()
        self.ensure_subtitle_dirs()
        
    def setup_subtitle_dirs(self) -> List[Path]:
        """Setup subtitle directory list."""
        dirs = []
        
        if "subtitle_dirs" in self.config:
            for dir_path in self.config["subtitle_dirs"]:
                try:
                    resolved_path = Path(dir_path).expanduser().resolve()
                    dirs.append(resolved_path)
                    logger.info(f"Added subtitle directory: {resolved_path}")
                except Exception as e:
                    logger.warning(f"Failed to resolve path {dir_path}: {e}")
        
        return dirs
        
    def ensure_subtitle_dirs(self):
        """Ensure all subtitle directories exist."""
        for subtitle_dir in self.subtitle_dirs:
            try:
                subtitle_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured subtitle directory exists: {subtitle_dir}")
            except Exception as e:
                logger.warning(f"Failed to create directory {subtitle_dir}: {e}")
        
    def find_subtitle_file(self, video_id: str) -> Optional[Path]:
        """Find subtitle file for a video ID."""
        logger.info(f"Looking for subtitle file, Video ID: {video_id}")
        
        priority_formats = [".ass", ".srt", ".vtt"]
        
        # 1. Exact match
        for ext in priority_formats:
            if ext in self.config["supported_formats"]:
                for i, subtitle_dir in enumerate(self.subtitle_dirs):
                    subtitle_file = subtitle_dir / f"{video_id}{ext}"
                    if subtitle_file.exists():
                        logger.info(f"✅ Found subtitle (Exact-{ext}): {subtitle_file}")
                        return subtitle_file
        
        # 2. Flexible match
        logger.info(f"No exact match found, trying flexible match...")
        
        for ext in priority_formats:
            if ext in self.config["supported_formats"]:
                for i, subtitle_dir in enumerate(self.subtitle_dirs):
                    if not subtitle_dir.exists():
                        continue
                    
                    try:
                        for subtitle_file in subtitle_dir.iterdir():
                            if subtitle_file.is_file() and subtitle_file.suffix.lower() == ext.lower():
                                filename = subtitle_file.stem
                                if (video_id in filename or 
                                    filename.endswith(f"-{video_id}") or 
                                    filename.endswith(f"_{video_id}") or
                                    filename.startswith(f"{video_id}-") or
                                    filename.startswith(f"{video_id}_")):
                                    logger.info(f"✅ Found subtitle (Flexible-{ext}): {subtitle_file}")
                                    return subtitle_file
                    except Exception as e:
                        logger.warning(f"Error scanning directory {subtitle_dir}: {e}")
        
        logger.info(f"❌ No matching subtitle file found")
        return None
        
    def get_subtitle_info(self, video_id: str) -> Optional[dict]:
        """Get subtitle file information."""
        subtitle_file = self.find_subtitle_file(video_id)
        if not subtitle_file:
            return None
            
        try:
            stat = subtitle_file.stat()
            return {
                "video_id": video_id,
                "filename": subtitle_file.name,
                "format": subtitle_file.suffix.lower(),
                "size": stat.st_size,
                "modified": int(stat.st_mtime),
                "path": str(subtitle_file)
            }
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            return None

def create_app(server: SubtitleServer) -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.route('/health')
    def health_check():
        return jsonify({
            "status": "ok",
            "service": "YouTube SubtitlePlus Server",
            "subtitle_dirs": [str(d) for d in server.subtitle_dirs],
            "supported_formats": server.config["supported_formats"]
        })

    @app.route('/subtitle/<video_id>')
    def get_subtitle(video_id):
        logger.info(f"Requesting subtitle: {video_id}")
        
        if not video_id or len(video_id) < 5:
            return jsonify({"error": "Invalid Video ID"}), 400
            
        subtitle_file = server.find_subtitle_file(video_id)
        if not subtitle_file:
            return jsonify({"error": "Subtitle file not found"}), 404
            
        try:
            # Try UTF-8 first
            try:
                with open(subtitle_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                encoding = 'utf-8'
            except UnicodeDecodeError:
                # Fallback to GBK
                with open(subtitle_file, 'r', encoding='gbk') as f:
                    content = f.read()
                encoding = 'gbk'
                
            info = server.get_subtitle_info(video_id)
            
            return jsonify({
                "success": True,
                "video_id": video_id,
                "content": content,
                "info": info,
                "encoding": encoding
            })
            
        except Exception as e:
            logger.error(f"Failed to read subtitle file: {e}")
            return jsonify({"error": "Failed to read file"}), 500

    @app.route('/subtitle/<video_id>/info')
    def get_subtitle_info(video_id):
        info = server.get_subtitle_info(video_id)
        if not info:
            return jsonify({"error": "Subtitle file not found"}), 404
            
        return jsonify({
            "success": True,
            "info": info
        })

    @app.route('/subtitle/process', methods=['POST'])
    def process_subtitle():
        """Process a subtitle file (save and translate)."""
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        video_id = data.get('video_id')
        content = data.get('content')
        target_lang = data.get('target_lang', 'zh') # Default to Chinese
        
        if not video_id or not content:
            return jsonify({"error": "Missing video_id or content"}), 400
            
        logger.info(f"Received processing request for video: {video_id}, target: {target_lang}")
        
        # 1. Determine output directory (use the first available directory)
        output_dir = None
        for d in server.subtitle_dirs:
            if d.exists():
                output_dir = d
                break
        
        if not output_dir:
            return jsonify({"error": "No valid subtitle directory found"}), 500
            
        # 2. Save SRT file
        srt_file = output_dir / f"{video_id}.srt"
        try:
            with open(srt_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved SRT file: {srt_file}")
        except Exception as e:
            logger.error(f"Failed to save SRT file: {e}")
            return jsonify({"error": f"Failed to save SRT file: {e}"}), 500
            
        # 3. Process in a separate thread to avoid blocking
        def run_processing(input_file, t_lang, out_dir):
            try:
                # Ensure environment is set up (logger, env vars)
                setup_environment(allow_missing_config=True) # Allow missing config to avoid exit(1) if something is wrong
                
                # Initialize service if needed (though process_single_file handles it)
                # We use default models for now
                process_single_file(
                    input_file=input_file,
                    target_lang=t_lang,
                    output_dir=out_dir,
                    model=DEFAULT_TRANSCRIPTION_MODEL,
                    llm_model=None, # Use default from env
                    model_precheck_passed=True # Skip check for SRT input
                )
                logger.info(f"Successfully processed video {video_id}")
            except Exception as e:
                logger.error(f"Error processing video {video_id}: {e}")

        # Start processing in background
        thread = threading.Thread(target=run_processing, args=(srt_file, target_lang, output_dir))
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Processing started",
            "file_path": str(srt_file)
        })

    @app.route('/list')
    def list_subtitles():
        subtitles = []
        processed_video_ids = set()
        
        try:
            for i, subtitle_dir in enumerate(server.subtitle_dirs):
                if not subtitle_dir.exists():
                    continue
                    
                for ext in server.config["supported_formats"]:
                    try:
                        for subtitle_file in subtitle_dir.glob(f"*{ext}"):
                            video_id = subtitle_file.stem
                            
                            if video_id not in processed_video_ids:
                                info = server.get_subtitle_info(video_id)
                                if info:
                                    info["source_dir"] = str(subtitle_dir)
                                    info["priority"] = i + 1
                                    subtitles.append(info)
                                    processed_video_ids.add(video_id)
                    except Exception as e:
                        logger.warning(f"Error scanning directory {subtitle_dir}: {e}")
                        
            return jsonify({
                "success": True,
                "count": len(subtitles),
                "subtitles": subtitles,
                "search_dirs": [str(d) for d in server.subtitle_dirs]
            })
            
        except Exception as e:
            logger.error(f"Failed to list subtitles: {e}")
            return jsonify({"error": "Failed to list files"}), 500

    @app.route('/config')
    def get_config():
        return jsonify({
            "success": True,
            "config": {
                "subtitle_dirs": [str(d) for d in server.subtitle_dirs],
                "supported_formats": server.config["supported_formats"],
                "server_port": server.config["server_port"],
                "server_host": server.config["server_host"]
            }
        })

    return app

def run_server(host: str, port: int, subtitle_dirs: List[str], debug: bool = False):
    """Run the subtitle server."""
    config = {
        "server_host": host,
        "server_port": port,
        "subtitle_dirs": subtitle_dirs
    }
    
    server = SubtitleServer(config)
    app = create_app(server)
    
    dirs_info = '\n'.join([f"║   {i+1}. {d}" + " " * (58 - len(str(d))) + "║" 
                          for i, d in enumerate(server.subtitle_dirs)])
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                YouTube SubtitlePlus Server                   ║
╠══════════════════════════════════════════════════════════════╣
║ Address: http://{host}:{port}                                ║
║ Subtitle Dirs:                                               ║
{dirs_info}
║ Formats: {', '.join(server.config['supported_formats'])}                    ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    app.run(host=host, port=port, debug=debug, threaded=True)
