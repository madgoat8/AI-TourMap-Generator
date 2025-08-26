#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小化流程，用于调试MPS设备上生成黑图的问题。
"""

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import numpy as np
import cv2
import os
import warnings

warnings.filterwarnings("ignore")

def run_minimal_test():
    """执行一个最简化的生成测试"""
    print("="*50)
    print("=== MPS 最小化流程调试开始 ===")
    print("="*50)

    # --- 1. 环境设置 ---
    if not torch.backends.mps.is_available():
        print("❌ MPS 设备不可用，测试无法进行。")
        return
    device = torch.device("mps")
    print(f"✅ 使用设备: {device}")

    # --- 2. 准备输入图像 ---
    input_image_path = "test1.png"
    if not os.path.exists(input_image_path):
        print(f"❌ 找不到输入文件 '{input_image_path}'，请确保它在当前目录中。")
        return
    
    print(f"\n--- 步骤 1: 准备输入图像 ---")
    image = Image.open(input_image_path).convert("RGB").resize((512, 512))
    canny_image = Image.fromarray(cv2.Canny(np.array(image), 100, 200)).convert("RGB")
    image.save("debug_minimal_input.png")
    canny_image.save("debug_minimal_canny.png")
    print("✅ 输入图和Canny线稿图已准备就绪 (debug_minimal_input.png, debug_minimal_canny.png)")

    # --- 3. 加载模型 ---
    print(f"\n--- 步骤 2: 加载模型 (使用 float32 精度) ---")
    try:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float32
            ),
            torch_dtype=torch.float32,
            safety_checker=None,
            cache_dir="./model_cache"
        ).to(device)
        print("✅ 模型加载完成。本次测试不进行任何attention processor修改，使用库的默认设置。")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # --- 4. 生成图像 ---
    print("\n--- 步骤 3: 开始生成图像 ---")
    prompt = "a photo of a cat"
    print(f"   - 使用最简单的提示词: '{prompt}'")
    generator = torch.Generator(device=device).manual_seed(0)
    
    try:
        with torch.no_grad():
            output_image = pipe(
                prompt,
                image=canny_image,
                generator=generator,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
        print("✅ 图像生成流程执行完毕。")
    except Exception as e:
        print(f"❌ 图像生成过程中发生错误: {e}")
        return

    # --- 5. 分析和保存结果 ---
    print("\n--- 步骤 4: 分析输出结果 ---")
    output_array = np.array(output_image)
    min_val, max_val, mean_val = output_array.min(), output_array.max(), output_array.mean()
    
    print(f"   - 像素值范围: {min_val} - {max_val}")
    print(f"   - 平均像素值: {mean_val:.2f}")
    
    if max_val < 10:
        print("   - 结论: 图像是黑色的。问题很可能出在模型计算或解码环节。")
    else:
        print("   - 结论: 图像不是黑色的。技术问题已解决，后续应专注于调整参数和提示词。 ")
        
    output_image.save("debug_minimal_output.png")
    print("✅ 最小化流程的输出图片已保存为 debug_minimal_output.png")
    print("="*50)
    print("=== 调试结束 ===")
    print("="*50)

if __name__ == "__main__":
    run_minimal_test()
