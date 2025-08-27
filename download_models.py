"""
模型预下载脚本 (国内镜像优化版)
用于提前下载所需的AI模型到本地缓存，支持离线运行
使用国内镜像源提升下载速度
"""

import os
import sys
from pathlib import Path

# 设置HuggingFace镜像源为国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置缓存目录
cache_dir = "./model_cache"
os.makedirs(cache_dir, exist_ok=True)

print("开始下载AI模型到本地缓存 (使用国内镜像)...")
print(f"镜像源: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
print(f"缓存目录: {os.path.abspath(cache_dir)}")

try:
    # 下载 ControlNet 模型
    print("\n1. 下载 ControlNet Canny 模型 (国内镜像)...")
    from diffusers import ControlNetModel
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        cache_dir=cache_dir,
        resume_download=True,  # 支持断点续传
        force_download=False   # 避免重复下载
    )
    print("✅ ControlNet 模型下载完成")

    # 下载 Stable Diffusion 模型
    print("\n2. 下载 Stable Diffusion v1.5 模型 (国内镜像)...")
    from diffusers import StableDiffusionControlNetPipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        cache_dir=cache_dir,
        safety_checker=None,
        resume_download=True,  # 支持断点续传
        force_download=False   # 避免重复下载
    )
    print("✅ Stable Diffusion 模型下载完成")

    # 下载 IP-Adapter 模型
    print("\n3. 下载 IP-Adapter 模型 (国内镜像)...")
    try:
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models", 
            weight_name="ip-adapter_sd15.bin",
            image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        print("✅ IP-Adapter 模型下载完成")
    except Exception as e:
        print(f"⚠️ IP-Adapter 下载警告: {e}")
        print("注意: IP-Adapter 可能需要在运行时首次加载")

    print("\n🎉 核心模型下载完成！")
    print("现在可以在离线环境中使用 start_offline.bat 启动服务")
    print("\n📊 下载统计:")
    print(f"缓存目录大小: {get_directory_size(cache_dir):.2f} MB")

except Exception as e:
    print(f"\n❌ 模型下载失败: {e}")
    print("\n🔧 故障排除建议:")
    print("1. 检查网络连接")
    print("2. 确认是否需要代理设置")
    print("3. 尝试重新运行脚本 (支持断点续传)")
    print("4. 如果国内镜像失效，可以手动设置其他镜像:")
    print("   export HF_ENDPOINT=https://hf-mirror.com")
    print("   或者: export HF_ENDPOINT=https://huggingface.co")
    sys.exit(1)

def get_directory_size(path):
    """计算目录大小 (MB)"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # 转换为MB
    except:
        return 0