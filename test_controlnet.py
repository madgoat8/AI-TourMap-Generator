import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image, ImageDraw
import numpy as np
import cv2

print("=== ControlNet 最简测试 ===")

# 检查设备
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")

# 创建一个简单的测试图像（白色背景上的黑色方框）
print("创建测试控制图像...")
test_image = Image.new("RGB", (512, 512), "white")
draw = ImageDraw.Draw(test_image)
draw.rectangle([100, 100, 400, 400], outline="black", width=5)
test_image.save("test_control_input.png")
print("测试控制图像已保存: test_control_input.png")

# 转换为Canny边缘
print("转换为Canny边缘...")
test_array = np.array(test_image)
gray = cv2.cvtColor(test_array, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(gray, 100, 200)
canny_image = Image.fromarray(canny).convert("RGB")
canny_image.save("test_canny.png")
print("Canny边缘图像已保存: test_canny.png")

try:
    print("加载ControlNet模型...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        cache_dir="./model_cache"
    )
    
    print("加载Stable Diffusion管道...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        cache_dir="./model_cache",
        safety_checker=None
    )
    
    if device == "mps":
        pipe.to(device)
        pipe.enable_attention_slicing()
    
    print("开始生成测试...")
    
    # 最简单的生成参数
    with torch.no_grad():
        if device == "mps":
            torch.mps.empty_cache()
        
        result = pipe(
            prompt="a red house",
            image=canny_image,
            num_inference_steps=10,  # 最少步数
            controlnet_conditioning_scale=0.5,  # 较低控制强度
            guidance_scale=7.5
        )
    
    # 保存结果
    output_image = result.images[0]
    output_image.save("test_output.png")
    print("✅ 测试成功！输出图像已保存: test_output.png")
    
    # 检查图像是否为纯黑
    output_array = np.array(output_image)
    if np.all(output_array == 0):
        print("❌ 警告：输出图像是纯黑色")
    else:
        print("✅ 输出图像包含内容")
        
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("=== 测试完成 ===")