import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

print("=== MPS黑色图像问题修复测试 ===")

# 强制使用MPS设备
if not torch.backends.mps.is_available():
    print("❌ MPS设备不可用，退出测试")
    exit()

device = "mps"
print(f"使用设备: {device}")

# 创建测试图像
print("创建测试控制图像...")
test_image = Image.new("RGB", (512, 512), "white")
draw = ImageDraw.Draw(test_image)
draw.rectangle([150, 150, 350, 350], outline="black", width=8)
draw.circle([256, 256], 50, outline="black", width=5)
test_image.save("mps_test_input.png")

# 转换为Canny边缘
test_array = np.array(test_image)
gray = cv2.cvtColor(test_array, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(gray, 100, 200)
canny_image = Image.fromarray(canny).convert("RGB")
canny_image.save("mps_test_canny.png")
print("测试图像已创建")

def test_mps_configuration(config_name, **kwargs):
    """测试不同的MPS配置"""
    print(f"\n--- 测试配置: {config_name} ---")
    
    try:
        # 清理MPS缓存
        torch.mps.empty_cache()
        
        # 加载模型
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=kwargs.get('dtype', torch.float16),
            cache_dir="./model_cache"
        )
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=kwargs.get('dtype', torch.float16),
            cache_dir="./model_cache",
            safety_checker=None
        )
        
        pipe.to(device)
        
        # 应用优化设置
        if kwargs.get('enable_attention_slicing', True):
            pipe.enable_attention_slicing()
        
        if kwargs.get('enable_memory_efficient_attention', False):
            pipe.enable_memory_efficient_attention()
            
        if kwargs.get('enable_vae_slicing', False):
            pipe.enable_vae_slicing()
        
        print(f"生成参数: {kwargs}")
        
        # 生成图像
        with torch.no_grad():
            torch.mps.empty_cache()
            
            # 设置随机种子确保可重现性
            generator = torch.Generator(device=device).manual_seed(kwargs.get('seed', 42))
            
            result = pipe(
                prompt=kwargs.get('prompt', "a colorful red house with blue windows"),
                negative_prompt=kwargs.get('negative_prompt', "black, dark, monochrome"),
                image=canny_image,
                num_inference_steps=kwargs.get('steps', 15),
                controlnet_conditioning_scale=kwargs.get('control_scale', 0.7),
                guidance_scale=kwargs.get('guidance_scale', 7.5),
                generator=generator
            )
        
        # 保存和分析结果
        output_image = result.images[0]
        filename = f"mps_test_{config_name.replace(' ', '_')}.png"
        output_image.save(filename)
        
        # 分析图像
        output_array = np.array(output_image)
        min_val, max_val = output_array.min(), output_array.max()
        mean_val = output_array.mean()
        
        print(f"✅ 生成成功: {filename}")
        print(f"   像素值范围: {min_val} - {max_val}")
        print(f"   平均像素值: {mean_val:.2f}")
        
        if max_val <= 10:
            print("   ❌ 图像几乎全黑")
            return False
        elif mean_val < 50:
            print("   ⚠️  图像偏暗")
            return False
        else:
            print("   ✅ 图像正常")
            return True
            
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False

# 测试不同配置
test_configs = [
    {
        'name': '默认配置',
        'dtype': torch.float16,
        'steps': 15,
        'control_scale': 0.7,
        'guidance_scale': 7.5,
        'seed': 42
    },
    {
        'name': 'float32精度',
        'dtype': torch.float32,
        'steps': 15,
        'control_scale': 0.7,
        'guidance_scale': 7.5,
        'seed': 42
    },
    {
        'name': '低控制强度',
        'dtype': torch.float16,
        'steps': 20,
        'control_scale': 0.5,
        'guidance_scale': 8.0,
        'seed': 42,
        'enable_vae_slicing': True
    },
    {
        'name': '高引导强度',
        'dtype': torch.float16,
        'steps': 25,
        'control_scale': 0.8,
        'guidance_scale': 12.0,
        'seed': 42,
        'enable_memory_efficient_attention': True
    },
    {
        'name': '优化提示词',
        'dtype': torch.float16,
        'steps': 20,
        'control_scale': 0.6,
        'guidance_scale': 9.0,
        'seed': 123,
        'prompt': "vibrant colorful building, bright red walls, blue sky, detailed architecture, high contrast",
        'negative_prompt': "black, dark, monochrome, low contrast, blurry, noise"
    },
    {
        'name': '最大优化',
        'dtype': torch.float32,
        'steps': 30,
        'control_scale': 0.6,
        'guidance_scale': 10.0,
        'seed': 456,
        'prompt': "bright colorful house, vivid colors, high saturation, detailed, sharp",
        'negative_prompt': "black, dark, monochrome, low quality, blurry, noise, artifacts",
        'enable_vae_slicing': True,
        'enable_memory_efficient_attention': True
    }
]

successful_configs = []

for config in test_configs:
    name = config.pop('name')
    success = test_mps_configuration(name, **config)
    if success:
        successful_configs.append(name)

print(f"\n=== 测试总结 ===")
print(f"成功配置数量: {len(successful_configs)}/{len(test_configs)}")
if successful_configs:
    print("成功的配置:")
    for config in successful_configs:
        print(f"  ✅ {config}")
    print("\n建议使用成功的配置更新主项目！")
else:
    print("❌ 所有MPS配置都失败，建议继续使用CPU版本")

print("=== 测试完成 ===")