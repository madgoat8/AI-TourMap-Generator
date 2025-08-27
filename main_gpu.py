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

# --- Background Task ---
def run_generation_task(job_id: str, coordinates: List[GeoCoordinate]):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(DEBUG_DIR, f"{timestamp}_{job_id[:8]}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"[Job {job_id}] 开始任务 (GPU优化版本)。调试文件保存在: {run_dir}")
    
    try:
        # 获取边界框详情
        bbox_details = get_bounding_box_details(coordinates)
        
        # 获取OSM数据
        logger.info(f"[Job {job_id}] 获取OSM数据...")
        osm_data = osm_client.fetch_data(bbox_details["bbox_str"])
        
        # 打印数据统计
        stats = osm_client.get_statistics(osm_data)
        logger.info(f"[Job {job_id}] OSM数据统计: {stats}")
        
        # 生成语义地图
        logger.info(f"[Job {job_id}] 生成语义地图...")
        semantic_image = semantic_generator.create_semantic_map(
            osm_data, bbox_details, os.path.join(run_dir, "semantic_map.png")
        )
        
        # Canny边缘检测
        logger.info(f"[Job {job_id}] 进行Canny边缘检测...")
        control_image_canny = canny_processor.process_semantic_map(
            semantic_image, os.path.join(run_dir, "control_canny.png")
        )
        
        # 加载固定的风格参考图
        logger.info(f"[Job {job_id}] 加载风格参考图 demo1.png...")
        style_image = None
        use_ip_adapter = False
        
        try:
            if hasattr(pipe, 'ip_adapter_available') and pipe.ip_adapter_available:
                style_image = Image.open("demo1.png").convert("RGB")
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

        # 根据IP-Adapter可用性调整提示词 - 使用配置中的提示词
        if use_ip_adapter:
            prompt = PromptConfig.PROMPT_WITH_IP_ADAPTER
            pipe.set_ip_adapter_scale(ModelConfig.IP_ADAPTER_SCALE)
            logger.info(f"[Job {job_id}] 使用IP-Adapter + ControlNet模式（优化版）")
        else:
            prompt = PromptConfig.PROMPT_WITHOUT_IP_ADAPTER
            logger.info(f"[Job {job_id}] 使用基础ControlNet模式（优化版）")
        
        # 使用配置中的负面提示词
        negative_prompt = PromptConfig.NEGATIVE_PROMPT
        
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
        
        # 数据清洗：检查并修复NaN值
        raw_image_np = output_images[0]
        if np.any(np.isnan(raw_image_np)):
            logger.warning(f"[Job {job_id}] 检测到NaN值，正在进行修复...")
            # 将NaN替换为0.5（中性灰色），避免产生黑点
            raw_image_np = np.nan_to_num(raw_image_np, nan=0.5)

        # 将清洗后的numpy数组转换为PIL Image
        final_image_np = (raw_image_np * 255).round().astype(np.uint8)
        final_image = Image.fromarray(final_image_np)

        final_image.save(os.path.join(run_dir, "final_map.png"))
        
        final_array = np.array(final_image)
        logger.info(f"[Job {job_id}] GPU优化生成完成！像素值范围: {final_array.min()} - {final_array.max()}")

        # GPU内存清理
        if device == "cuda":
            torch.cuda.empty_cache()

        buffered = BytesIO()
        final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info(f"[Job {job_id}] 任务完成。")
        JOBS[job_id] = {
            "status": "complete",
            "result": {
                "image_base64": img_str,
                "bounds": [[bbox_details["min_lat"], bbox_details["min_lng"]], [bbox_details["max_lat"], bbox_details["max_lng"]]]
            }}
    except Exception as e:
        logger.error(f"[Job {job_id}] 任务失败: {e}", exc_info=True)
        JOBS[job_id] = {"status": "failed", "error": str(e)}
        # 出错时也清理GPU内存
        if device == "cuda":
            torch.cuda.empty_cache()

# --- API Endpoints ---
@app.get("/", response_class=FileResponse)
async def read_index():
    return "index.html"

@app.post("/api/start_generation")
def start_generation(area: AreaSelection, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "processing"}
    background_tasks.add_task(run_generation_task, job_id, area.coordinates)
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
