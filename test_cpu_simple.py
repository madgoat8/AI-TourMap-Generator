import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageDraw
import numpy as np
import cv2

print("=== CPU模式 ControlNet 测试 ===")

# 强制使用CPU
device = "cpu"
print(f"强制使用设备: {device}")

# 创建简单测试图像
print("创建测试控制图像...")
test_image = Image.new("RGB", (512, 512), "white")
draw = ImageDraw.Draw(test_image)
draw.rectangle([150, 150, 350, 350], outline="black", width=10)
test_image.save("cpu_test_input.png")

# 转换为Canny
test_array = np.array(test_image)
gray = cv2.cvtColor(test_array, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(gray, 50, 150)
canny_image = Image.fromarray(canny).convert("RGB")
canny_image.save("cpu_test_canny.png")
print("测试图像已创建")

try:
    print("加载ControlNet模型 (CPU模式)...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float32,
        cache_dir="./model_cache"
    )
    
    print("加载Stable Diffusion管道...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32,
        cache_dir="./model_cache",
        safety_checker=None
    )
    
    print("开始生成 (CPU模式，会比较慢)...")
    
    result = pipe(
        prompt="a simple red square building",
        image=canny_image,
        num_inference_steps=5,  # 极少步数用于快速测试
        controlnet_conditioning_scale=0.8,
        guidance_scale=7.5
    )
    
    output_image = result.images[0]
    output_image.save("cpu_test_output.png")
    print("✅ CPU测试完成！输出: cpu_test_output.png")
    
    # 检查是否为黑色
    output_array = np.array(output_image)
    if np.all(output_array < 10):  # 几乎全黑
        print("❌ 输出图像几乎是纯黑色")
    else:
        print("✅ 输出图像包含内容")
        print(f"像素值范围: {output_array.min()} - {output_array.max()}")
        
except Exception as e:
    print(f"❌ CPU测试失败: {e}")
    import traceback
    traceback.print_exc()

print("=== CPU测试完成 ===")