#!/usr/bin/env python3
"""诊断转录模型缓存状态"""

import os
from pathlib import Path
import json
import hashlib

def check_cache_status():
    """检查所有可能的缓存位置"""

    print("=" * 60)
    print("🔍 转录模型缓存状态诊断")
    print("=" * 60)

    # 获取缓存根目录
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or Path.home() / ".cache" / "huggingface"
    cache_dir = Path(cache_dir)

    print(f"\n📁 缓存根目录: {cache_dir}")
    print(f"   存在: {'✅' if cache_dir.exists() else '❌'}")

    # 检查1: 标准HF缓存
    print("\n" + "=" * 60)
    print("1️⃣  标准 Hugging Face 缓存")
    print("=" * 60)

    model_ids = [
        "nvidia/parakeet-tdt-0.6b",
        "nvidia/parakeet-tdt-1.1b"
    ]

    for model_id in model_ids:
        print(f"\n🤖 模型: {model_id}")
        model_cache_name = model_id.replace("/", "--")
        model_cache_dir = cache_dir / "hub" / f"models--{model_cache_name}"

        print(f"   缓存目录: {model_cache_dir}")
        if model_cache_dir.exists():
            print("   ✅ 目录存在")

            # 检查 snapshots
            snapshots_dir = model_cache_dir / "snapshots"
            if snapshots_dir.exists():
                snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                if snapshot_dirs:
                    print(f"   ✅ 找到 {len(snapshot_dirs)} 个快照")

                    # 检查最新快照的文件
                    latest_snapshot = max(snapshot_dirs, key=lambda d: d.stat().st_mtime)
                    print(f"   📦 最新快照: {latest_snapshot.name}")

                    config_file = latest_snapshot / "config.json"
                    weight_file = latest_snapshot / "model.safetensors"

                    print(f"   {'✅' if config_file.exists() else '❌'} config.json ({config_file.stat().st_size / 1024:.1f} KB)" if config_file.exists() else "   ❌ config.json 不存在")
                    print(f"   {'✅' if weight_file.exists() else '❌'} model.safetensors ({weight_file.stat().st_size / (1024*1024):.1f} MB)" if weight_file.exists() else "   ❌ model.safetensors 不存在")
                else:
                    print("   ❌ 没有找到快照")
            else:
                print("   ❌ snapshots 目录不存在")
        else:
            print("   ❌ 目录不存在")

    # 检查2: 存储优化缓存
    print("\n" + "=" * 60)
    print("2️⃣  存储优化缓存 (optimized_models)")
    print("=" * 60)

    optimized_cache_dir = cache_dir / "optimized_models"
    print(f"\n📁 优化缓存目录: {optimized_cache_dir}")
    print(f"   存在: {'✅' if optimized_cache_dir.exists() else '❌'}")

    if optimized_cache_dir.exists():
        cache_dirs = [d for d in optimized_cache_dir.iterdir() if d.is_dir()]
        print(f"   找到 {len(cache_dirs)} 个优化缓存")

        for cache_dir_item in cache_dirs:
            metadata_file = cache_dir_item / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                print(f"\n   📦 缓存: {cache_dir_item.name}")
                print(f"      模型: {metadata.get('model_id', 'Unknown')}")
                print(f"      数据类型: {metadata.get('dtype', 'Unknown')}")

                # 检查文件完整性
                config_file = cache_dir_item / "config.json"
                weight_file = cache_dir_item / "optimized_weights.safetensors"

                print(f"      {'✅' if config_file.exists() else '❌'} config.json")
                print(f"      {'✅' if weight_file.exists() else '❌'} optimized_weights.safetensors")

    # 检查3: 计算预期的缓存键
    print("\n" + "=" * 60)
    print("3️⃣  预期的缓存键")
    print("=" * 60)

    for model_id in model_ids:
        for dtype_str in ["bfloat16", "float16", "float32"]:
            content = f"{model_id}_{dtype_str}"
            cache_key = hashlib.md5(content.encode()).hexdigest()
            print(f"\n   {model_id} ({dtype_str})")
            print(f"   缓存键: {cache_key}")

            expected_dir = optimized_cache_dir / cache_key
            print(f"   预期路径: {expected_dir}")
            print(f"   存在: {'✅' if expected_dir.exists() else '❌'}")

    print("\n" + "=" * 60)
    print("✅ 诊断完成")
    print("=" * 60)

if __name__ == "__main__":
    check_cache_status()
