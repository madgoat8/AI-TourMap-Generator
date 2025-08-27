from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict
import requests
from PIL import Image, ImageDraw
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import cv2
import os
import uuid
import base64
import json
from io import BytesIO
import webbrowser
import logging
import datetime
import warnings

# 导入配置和工具模块
from config import (
    ModelConfig, PromptConfig, GPUConfig, LogConfig,
    DEBUG_DIR, CACHE_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_MAP, OVERPASS_URL,
    SERVER_HOST, SERVER_PORT, ensure_directories, setup_environment
)
from utils import GeoCoordinate, get_bounding_box_details
from data import OSMClient
from image_processing import SemanticMapGenerator, CannyProcessor
from models import ModelLoader

# 设置环境和目录
setup_environment()
ensure_directories()

# 完全抑制 xFormers 和 Triton 相关的警告和输出
warnings.filterwarnings("ignore")

# --- Configuration & Global State ---
logging.basicConfig(
    level=getattr(logging, LogConfig.LEVEL),
    format=LogConfig.FORMAT,
    filename=LogConfig.FILENAME,
    filemode=LogConfig.FILEMODE
)
logger = logging.getLogger(__name__)

JOBS: Dict[str, Dict] = {}
HAS_OPENED_BROWSER = False

# --- Pydantic Models ---
class AreaSelection(BaseModel):
    coordinates: List[GeoCoordinate]
    screenshot_base64: str = None  # 可选的原始截图

# --- FastAPI App with Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global HAS_OPENED_BROWSER
    if not HAS_OPENED_BROWSER:
        try:
            url = f"http://{SERVER_HOST}:{SERVER_PORT}"
            logger.info(f"GPU版本服务启动。打开浏览器: {url}")
            webbrowser.open(url)
            HAS_OPENED_BROWSER = True
        except Exception as e:
            logger.error(f"无法打开浏览器: {e}")
    
    yield
    
    # Shutdown (if needed)
    pass

app = FastAPI(lifespan=lifespan)

# --- AI Model Loading --- 
logger.info("初始化AI模型加载器...")
model_loader = ModelLoader()

# 加载模型
logger.info("加载AI模型 (GPU优化版配置)...")
controlnet = model_loader.load_controlnet()
pipe = model_loader.load_stable_diffusion_pipeline()

# 设置GPU优化
model_loader.setup_gpu_optimizations()

# 加载IP-Adapter
model_loader.load_ip_adapter()

# 获取设备信息
device = model_loader.device
dtype = model_loader.dtype

logger.info("AI模型加载完成。")

# --- 初始化处理器 ---
osm_client = OSMClient()
semantic_generator = SemanticMapGenerator()
canny_processor = CannyProcessor()

# --- 调试辅助函数 ---
def save_debug_info(run_dir: str, step_name: str, data, data_type: str = "json"):
    """保存调试信息的通用函数"""
    try:
        if data_type == "json":
            with open(os.path.join(run_dir, f"{step_name}.json"), "w", encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif data_type == "image":
            if hasattr(data, 'save'):
                data.save(os.path.join(run_dir, f"{step_name}.png"))
        elif data_type == "text":
            with open(os.path.join(run_dir, f"{step_name}.txt"), "w", encoding='utf-8') as f:
                f.write(str(data))
        logger.info(f"调试信息已保存: {step_name}")
    except Exception as e:
        logger.error(f"保存调试信息失败 {step_name}: {e}")

def create_process_comparison(semantic_image: Image.Image, canny_image: Image.Image, final_image: Image.Image, run_dir: str, original_screenshot: Image.Image = None):
    """生成处理流程对比图"""
    try:
        # 确保所有图像尺寸一致
        target_size = (400, 400)  # 稍微缩小以容纳更多图像
        semantic_resized = semantic_image.resize(target_size, Image.Resampling.LANCZOS)
        canny_resized = canny_image.resize(target_size, Image.Resampling.LANCZOS)
        final_resized = final_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 检查是否有原始截图
        if original_screenshot:
            original_resized = original_screenshot.resize(target_size, Image.Resampling.LANCZOS)
            # 创建四联对比图
            comparison_width = target_size[0] * 4 + 50  # 添加间隔
            comparison_height = target_size[1] + 60  # 添加标题空间
            
            comparison = Image.new('RGB', (comparison_width, comparison_height), color='white')
            
            # 添加标题
            from PIL import ImageFont, ImageDraw
            draw = ImageDraw.Draw(comparison)
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), "原始截图", fill='black', font=font)
            draw.text((target_size[0] + 25, 10), "语义地图", fill='black', font=font)
            draw.text((target_size[0] * 2 + 35, 10), "Canny边缘", fill='black', font=font)
            draw.text((target_size[0] * 3 + 45, 10), "AI生成结果", fill='black', font=font)
            
            # 粘贴图像
            comparison.paste(original_resized, (10, 40))
            comparison.paste(semantic_resized, (target_size[0] + 15, 40))
            comparison.paste(canny_resized, (target_size[0] * 2 + 25, 40))
            comparison.paste(final_resized, (target_size[0] * 3 + 35, 40))
        else:
            # 创建三联对比图（原有逻辑）
            comparison_width = target_size[0] * 3 + 40  # 添加间隔
            comparison_height = target_size[1] + 60  # 添加标题空间
            
            comparison = Image.new('RGB', (comparison_width, comparison_height), color='white')
            
            # 添加标题
            from PIL import ImageFont, ImageDraw
            draw = ImageDraw.Draw(comparison)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), "语义地图", fill='black', font=font)
            draw.text((target_size[0] + 25, 10), "Canny边缘", fill='black', font=font)
            draw.text((target_size[0] * 2 + 40, 10), "AI生成结果", fill='black', font=font)
            
            # 粘贴图像
            comparison.paste(semantic_resized, (10, 40))
            comparison.paste(canny_resized, (target_size[0] + 20, 40))
            comparison.paste(final_resized, (target_size[0] * 2 + 30, 40))
        
        comparison.save(os.path.join(run_dir, "09_process_comparison.png"))
        logger.info("处理流程对比图生成完成")
    except Exception as e:
        logger.error(f"生成处理流程对比图失败: {e}")

# --- Background Task ---
def run_generation_task(job_id: str, coordinates: List[GeoCoordinate], screenshot_base64: str = None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(DEBUG_DIR, f"{timestamp}_{job_id[:8]}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"[Job {job_id}] 开始任务 (GPU优化版本)。调试文件保存在: {run_dir}")
    
    try:
        # 1. 保存原始截图（如果提供）
        original_screenshot = None
        if screenshot_base64:
            try:
                # 解码base64图像
                image_data = base64.b64decode(screenshot_base64)
                original_screenshot = Image.open(BytesIO(image_data)).convert("RGB")
                original_screenshot.save(os.path.join(run_dir, "00_original_screenshot.png"))
                logger.info(f"[Job {job_id}] 原始截图已保存")
            except Exception as e:
                logger.warning(f"[Job {job_id}] 保存原始截图失败: {e}")
        
        # 2. 保存坐标信息
        coords_info = {
            "coordinates": [{"lat": c.lat, "lng": c.lng} for c in coordinates],
            "timestamp": timestamp,
            "job_id": job_id,
            "description": "Web UI生成任务",
            "has_original_screenshot": screenshot_base64 is not None
        }
        save_debug_info(run_dir, "01_coordinates", coords_info, "json")
        
        # 2. 获取边界框详情
        bbox_details = get_bounding_box_details(coordinates)
        save_debug_info(run_dir, "02_bbox_info", bbox_details, "json")
        logger.info(f"[Job {job_id}] 边界框: {bbox_details['bbox_str']}")
        
        # 3. 获取OSM数据
        logger.info(f"[Job {job_id}] 获取OSM数据...")
        osm_data = osm_client.fetch_data(bbox_details["bbox_str"])
        
        # 4. 打印并保存数据统计
        stats = osm_client.get_statistics(osm_data)
        save_debug_info(run_dir, "03_osm_stats", stats, "json")
        logger.info(f"[Job {job_id}] OSM数据统计: {stats}")
        
        # 5. 生成语义地图
        logger.info(f"[Job {job_id}] 生成语义地图...")
        semantic_image = semantic_generator.create_semantic_map(
            osm_data, bbox_details, os.path.join(run_dir, "04_semantic_map.png")
        )
        
        # 6. Canny边缘检测
        logger.info(f"[Job {job_id}] 进行Canny边缘检测...")
        control_image_canny = canny_processor.process_semantic_map(
            semantic_image, os.path.join(run_dir, "05_control_canny.png")
        )
        
        # 7. 加载固定的风格参考图
        logger.info(f"[Job {job_id}] 加载风格参考图 demo1.png...")
        style_image = None
        use_ip_adapter = False
        
        try:
            if hasattr(pipe, 'ip_adapter_available') and pipe.ip_adapter_available:
                style_image = Image.open("demo1.png").convert("RGB")
                # 保存风格参考图到调试目录
                style_image.save(os.path.join(run_dir, "06_style_reference.png"))
                use_ip_adapter = True
                logger.info(f"[Job {job_id}] 风格参考图加载成功，将使用IP-Adapter")
            else:
                logger.info(f"[Job {job_id}] IP-Adapter不可用，使用基础ControlNet模式")
        except FileNotFoundError:
            logger.warning(f"[Job {job_id}] 风格参考图 demo1.png 未找到，将使用基础模式")
            use_ip_adapter = False
        except Exception as e:
            logger.warning(f"[Job {job_id}] 加载风格图片失败: {e}，使用基础模式")
            use_ip_adapter = False

        # 8. 根据IP-Adapter可用性调整提示词 - 使用配置中的提示词
        if use_ip_adapter:
            prompt = PromptConfig.PROMPT_WITH_IP_ADAPTER
            pipe.set_ip_adapter_scale(ModelConfig.IP_ADAPTER_SCALE)
            logger.info(f"[Job {job_id}] 使用IP-Adapter + ControlNet模式（优化版）")
        else:
            prompt = PromptConfig.PROMPT_WITHOUT_IP_ADAPTER
            logger.info(f"[Job {job_id}] 使用基础ControlNet模式（优化版）")
        
        # 使用配置中的负面提示词
        negative_prompt = PromptConfig.NEGATIVE_PROMPT
        
        # 9. 保存生成参数
        generation_params_info = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "use_ip_adapter": use_ip_adapter,
            "device": str(device),
            "dtype": str(dtype),
            "inference_steps": ModelConfig.INFERENCE_STEPS_GPU if device == "cuda" else ModelConfig.INFERENCE_STEPS_CPU,
            "controlnet_scale": ModelConfig.CONTROLNET_SCALE,
            "guidance_scale": ModelConfig.GUIDANCE_SCALE,
            "random_seed": ModelConfig.RANDOM_SEED,
            "ip_adapter_scale": ModelConfig.IP_ADAPTER_SCALE if use_ip_adapter else None
        }
        save_debug_info(run_dir, "07_generation_params", generation_params_info, "json")
        
        logger.info(f"[Job {job_id}] 开始AI生成 ({'IP-Adapter + ControlNet' if use_ip_adapter else 'ControlNet'}, GPU优化加速)...")
        
        # GPU内存管理
        if device == "cuda":
            torch.cuda.empty_cache()
        
        generator = torch.Generator(device=device).manual_seed(ModelConfig.RANDOM_SEED)
        
        # 使用配置中的推理参数
        inference_steps = ModelConfig.INFERENCE_STEPS_GPU if device == "cuda" else ModelConfig.INFERENCE_STEPS_CPU
        
        # 使用推理模式和混合精度优化
        with torch.inference_mode():
            if device == "cuda" and dtype == torch.float16:
                # 使用自动混合精度加速
                with torch.cuda.amp.autocast():
                    generation_params = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "image": control_image_canny,
                        "num_inference_steps": inference_steps,
                        "controlnet_conditioning_scale": ModelConfig.CONTROLNET_SCALE,
                        "guidance_scale": ModelConfig.GUIDANCE_SCALE,
                        "generator": generator,
                        "output_type": "np"
                    }
                    
                    if use_ip_adapter and style_image is not None:
                        generation_params["ip_adapter_image"] = style_image
                    
                    output_images = pipe(**generation_params).images
            else:
                # CPU模式或float32模式
                generation_params = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": control_image_canny,
                    "num_inference_steps": inference_steps,
                    "controlnet_conditioning_scale": ModelConfig.CONTROLNET_SCALE,
                    "guidance_scale": ModelConfig.GUIDANCE_SCALE,
                    "generator": generator,
                    "output_type": "np"
                }
                
                if use_ip_adapter and style_image is not None:
                    generation_params["ip_adapter_image"] = style_image
                
                output_images = pipe(**generation_params).images
        
        # 10. 数据清洗：检查并修复NaN值
        raw_image_np = output_images[0]
        if np.any(np.isnan(raw_image_np)):
            logger.warning(f"[Job {job_id}] 检测到NaN值，正在进行修复...")
            # 将NaN替换为0.5（中性灰色），避免产生黑点
            raw_image_np = np.nan_to_num(raw_image_np, nan=0.5)

        # 11. 将清洗后的numpy数组转换为PIL Image
        final_image_np = (raw_image_np * 255).round().astype(np.uint8)
        final_image = Image.fromarray(final_image_np)

        final_image.save(os.path.join(run_dir, "08_final_map.png"))
        
        final_array = np.array(final_image)
        logger.info(f"[Job {job_id}] GPU优化生成完成！像素值范围: {final_array.min()} - {final_array.max()}")

        # 12. 生成处理流程对比图
        create_process_comparison(semantic_image, control_image_canny, final_image, run_dir, original_screenshot)
        
        # 13. 保存详细的生成日志
        generation_log = f"""
Web UI生成完成报告
==================
任务ID: {job_id}
时间戳: {timestamp}
坐标区域: {len(coordinates)}个坐标点
边界框: {bbox_details['bbox_str']}
OSM数据统计: {stats}
使用模式: {'IP-Adapter + ControlNet' if use_ip_adapter else 'ControlNet'}
设备: {device}
数据类型: {dtype}
推理步数: {ModelConfig.INFERENCE_STEPS_GPU if device == "cuda" else ModelConfig.INFERENCE_STEPS_CPU}
ControlNet缩放: {ModelConfig.CONTROLNET_SCALE}
引导缩放: {ModelConfig.GUIDANCE_SCALE}
随机种子: {ModelConfig.RANDOM_SEED}
最终图像尺寸: {final_image.size}
像素值范围: {final_array.min()} - {final_array.max()}
生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        save_debug_info(run_dir, "10_generation_log", generation_log, "text")
        
        # 14. 保存任务摘要
        task_summary = {
            "job_id": job_id,
            "timestamp": timestamp,
            "status": "completed",
            "coordinates_count": len(coordinates),
            "bbox": bbox_details['bbox_str'],
            "osm_stats": stats,
            "generation_mode": "IP-Adapter + ControlNet" if use_ip_adapter else "ControlNet",
            "device": str(device),
            "inference_steps": ModelConfig.INFERENCE_STEPS_GPU if device == "cuda" else ModelConfig.INFERENCE_STEPS_CPU,
            "final_image_size": final_image.size,
            "pixel_range": {"min": int(final_array.min()), "max": int(final_array.max())},
            "debug_directory": run_dir
        }
        save_debug_info(run_dir, "11_task_summary", task_summary, "json")

        # 15. GPU内存清理
        if device == "cuda":
            torch.cuda.empty_cache()

        buffered = BytesIO()
        final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info(f"[Job {job_id}] 任务完成。完整调试信息已保存到: {run_dir}")
        JOBS[job_id] = {
            "status": "complete",
            "result": {
                "image_base64": img_str,
                "bounds": [[bbox_details["min_lat"], bbox_details["min_lng"]], [bbox_details["max_lat"], bbox_details["max_lng"]]],
                "debug_info": {
                    "job_id": job_id,
                    "debug_directory": run_dir,
                    "generation_mode": "IP-Adapter + ControlNet" if use_ip_adapter else "ControlNet",
                    "osm_stats": stats
                }
            }}
    except Exception as e:
        logger.error(f"[Job {job_id}] 任务失败: {e}", exc_info=True)
        
        # 保存错误信息到调试目录
        error_info = {
            "error": str(e),
            "job_id": job_id,
            "timestamp": timestamp,
            "error_type": type(e).__name__,
            "debug_directory": run_dir
        }
        save_debug_info(run_dir, "error_log", error_info, "json")
        
        # 保存错误详情文本
        error_details = f"""
任务执行错误报告
================
任务ID: {job_id}
时间戳: {timestamp}
错误类型: {type(e).__name__}
错误信息: {str(e)}
调试目录: {run_dir}
发生时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        save_debug_info(run_dir, "error_details", error_details, "text")
        
        JOBS[job_id] = {
            "status": "failed", 
            "error": str(e),
            "debug_info": {
                "job_id": job_id,
                "debug_directory": run_dir,
                "error_type": type(e).__name__
            }
        }
        
        # 出错时也清理GPU内存
        if 'device' in locals() and device == "cuda":
            torch.cuda.empty_cache()

# --- API Endpoints ---
@app.get("/", response_class=FileResponse)
async def read_index():
    return "index.html"

@app.post("/api/start_generation")
def start_generation(area: AreaSelection, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "processing"}
    background_tasks.add_task(run_generation_task, job_id, area.coordinates, area.screenshot_base64)
    return {"job_id": job_id}

@app.get("/api/job_status/{job_id}")
def get_job_status(job_id: str):
    return JOBS.get(job_id, {"status": "not_found"})

# --- GPU状态监控API ---
@app.get("/api/gpu_status")
def get_gpu_status():
    return model_loader.get_device_info()

# --- Main Execution (for direct run) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("直接运行GPU优化版本脚本，启动uvicorn服务...")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
