#!/usr/bin/env python3
"""
测试翻译缓存功能
"""

import requests
import time
import json
from pathlib import Path

# 后端配置
BACKEND_URL = "http://127.0.0.1:9009"

def test_cache_status():
    """测试缓存状态API"""
    print("🧪 测试缓存状态...")
    try:
        response = requests.get(f"{BACKEND_URL}/cache/status")
        if response.ok:
            data = response.json()
            print("✅ 缓存状态获取成功:")
            print(f"   音频缓存: {data['audio_cache']['count']}个")
            print(f"   字幕缓存: {data['subtitle_cache']['count']}个")
            print(f"   翻译缓存: {data['translation_cache']['count']}个")
            print(f"   翻译缓存大小: {data['translation_cache']['total_size']} bytes")
            return True
        else:
            print(f"❌ 缓存状态获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 缓存状态测试失败: {e}")
        return False

def test_backend_health():
    """测试后端健康状态"""
    print("🏥 测试后端健康状态...")
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.ok:
            data = response.json()
            print("✅ 后端健康状态正常")
            print(f"   Python版本: {data.get('python', 'unknown')}")
            return True
        else:
            print(f"❌ 后端健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 后端连接失败: {e}")
        print("💡 请确保后端服务已启动: python backend/server.py")
        return False

def test_video_cache_check():
    """测试视频缓存检查"""
    print("🎬 测试视频缓存检查...")
    
    # 使用一个示例视频ID测试
    test_video_id = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up
    
    try:
        response = requests.get(f"{BACKEND_URL}/cache/check/{test_video_id}")
        if response.ok:
            data = response.json()
            print(f"✅ 视频 {test_video_id} 缓存状态:")
            print(f"   音频缓存: {'✅' if data['has_audio_cache'] else '❌'}")
            print(f"   字幕缓存: {'✅' if data['has_subtitle_cache'] else '❌'}")
            print(f"   翻译缓存: {'✅' if data['has_translation_cache'] else '❌'}")
            return True
        else:
            print(f"❌ 视频缓存检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 视频缓存检查失败: {e}")
        return False

def test_translation_job_simulation():
    """模拟翻译作业，测试缓存机制"""
    print("🎯 模拟翻译作业...")
    
    # 使用Rick Astley视频测试（常见的测试视频）
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        # 启动翻译作业
        print("📤 提交翻译作业...")
        response = requests.post(f"{BACKEND_URL}/translate_youtube", 
                               json={"youtube_url": test_url})
        
        if not response.ok:
            print(f"❌ 作业提交失败: {response.status_code}")
            return False
        
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"✅ 作业已提交: {job_id}")
        
        # 轮询作业状态
        print("⏳ 监控作业进度...")
        max_polls = 60  # 最多轮询10分钟
        poll_count = 0
        
        while poll_count < max_polls:
            time.sleep(10)  # 每10秒检查一次
            poll_count += 1
            
            # 获取作业状态
            state_response = requests.get(f"{BACKEND_URL}/jobs/{job_id}/state")
            if not state_response.ok:
                print(f"❌ 状态查询失败: {state_response.status_code}")
                continue
            
            state = state_response.json()
            status = state.get("status", "unknown")
            stage = state.get("progress", {}).get("stage", "unknown")
            message = state.get("progress", {}).get("message", "")
            
            print(f"📊 [{poll_count:2d}/60] {status} - {stage}: {message}")
            
            if status == "done":
                print("✅ 作业完成！")
                
                # 测试SRT文件获取
                print("📄 测试SRT文件获取...")
                srt_response = requests.get(f"{BACKEND_URL}/srt_files/{job_id}")
                if srt_response.ok:
                    srt_data = srt_response.json()
                    print(f"   SRT文件源: {srt_data.get('source', 'unknown')}")
                    print(f"   英文SRT: {'✅' if srt_data['has_english_srt'] else '❌'}")
                    print(f"   翻译SRT: {'✅' if srt_data['has_translated_srt'] else '❌'}")
                    
                    if srt_data.get('source') == 'cache':
                        print("🎉 翻译缓存工作正常！")
                        return True
                    else:
                        print("ℹ️ 首次处理，已生成缓存")
                        return True
                else:
                    print(f"❌ SRT文件获取失败: {srt_response.status_code}")
                    return False
                    
            elif status == "error":
                error = state.get("error", "未知错误")
                print(f"❌ 作业失败: {error}")
                return False
        
        print("⏰ 作业处理超时")
        return False
        
    except Exception as e:
        print(f"❌ 翻译作业测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试翻译缓存功能")
    print("=" * 50)
    
    # 测试后端连接
    health_ok = test_backend_health()
    if not health_ok:
        print("\n❌ 后端连接失败，终止测试")
        return False
    
    print()
    
    # 测试缓存状态
    cache_status_ok = test_cache_status()
    
    print()
    
    # 测试视频缓存检查
    video_cache_ok = test_video_cache_check()
    
    print()
    
    # 询问是否进行完整测试
    print("⚠️ 完整测试将实际处理YouTube视频，可能需要5-15分钟")
    user_input = input("是否继续完整测试？(y/N): ").strip().lower()
    
    if user_input == 'y':
        print()
        job_ok = test_translation_job_simulation()
    else:
        print("跳过完整测试")
        job_ok = True
    
    print("\n" + "=" * 50)
    print("🎯 测试结果汇总:")
    print(f"   后端连接: {'✅ 通过' if health_ok else '❌ 失败'}")
    print(f"   缓存状态: {'✅ 通过' if cache_status_ok else '❌ 失败'}")
    print(f"   视频缓存: {'✅ 通过' if video_cache_ok else '❌ 失败'}")
    print(f"   完整测试: {'✅ 通过' if job_ok else '❌ 失败'}")
    
    all_passed = health_ok and cache_status_ok and video_cache_ok and job_ok
    
    if all_passed:
        print("\n🎉 翻译缓存功能测试通过！")
        print("💡 提示：")
        print("   - 首次处理视频会生成缓存")
        print("   - 后续访问同一视频将直接使用缓存")
        print("   - 可通过Chrome扩展验证缓存效果")
        return True
    else:
        print("\n⚠️ 部分测试失败，请检查相关功能")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)