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

# 完全抑制 xFormers 和 Triton 相关的警告和输出
warnings.filterwarnings("ignore")
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

# --- Configuration & Global State ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    filename='generation_gpu.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

DEBUG_DIR = "debug_runs"
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

cache_dir = "./model_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

JOBS: Dict[str, Dict] = {}
HAS_OPENED_BROWSER = False

# --- Pydantic Models ---
class GeoCoordinate(BaseModel):
    lat: float
    lng: float

class AreaSelection(BaseModel):
    coordinates: List[GeoCoordinate]

# --- FastAPI App with Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global HAS_OPENED_BROWSER
    if not HAS_OPENED_BROWSER:
        try:
            url = "http://127.0.0.1:8003"
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
# GPU优化配置
logger.info("加载AI模型 (GPU优化版配置)...")

# GPU设备检测和配置
if torch.cuda.is_available():
    device = "cuda"
    # GPU内存优化
    torch.cuda.empty_cache()
    # 启用优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    logger.info(f"使用GPU设备: {torch.cuda.get_device_name()}")
    logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = "cpu"
    logger.warning("GPU不可用，使用CPU运行")

logger.info(f"使用设备: {device}")

# GPU优化：使用float16精度提升性能
dtype = torch.float16 if device == "cuda" else torch.float32
if device == "cuda":
    logger.info("GPU模式：使用float16精度提升性能")
else:
    logger.info("CPU模式：使用float32精度")

# 静默加载模型，抑制所有警告
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    logger.info(f"加载ControlNet模型 (dtype: {dtype})...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", 
        torch_dtype=dtype,
        cache_dir=cache_dir
    )

    logger.info(f"加载Stable Diffusion管道 (dtype: {dtype})...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet, 
        torch_dtype=dtype,
        cache_dir=cache_dir,
        safety_checker=None
    ).to(device)

# GPU专用优化
if device == "cuda":
    # 启用内存高效注意力机制
    pipe.enable_attention_slicing()
    logger.info("✅ 已启用注意力切片优化")
    
    # 尝试启用xFormers优化（Windows平台可能不支持）
    xformers_available = False
    try:
        # 静默检查xFormers可用性
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import xformers
            # 尝试启用xFormers优化
            pipe.enable_xformers_memory_efficient_attention()
            xformers_available = True
            logger.info("✅ 已启用xFormers内存高效注意力机制")
    except (ImportError, ModuleNotFoundError):
        logger.info("ℹ️  xFormers未安装，使用标准注意力机制")
    except Exception:
        # 静默处理所有xFormers相关错误
        logger.info("ℹ️  xFormers在当前环境不可用，使用标准注意力机制")
    
    # 启用VAE切片以节省内存
    try:
        pipe.enable_vae_slicing()
        logger.info("✅ 已启用VAE切片优化")
    except Exception as e:
        logger.warning(f"VAE切片优化失败: {e}")
    
    # 启用VAE平铺（对大图像有效）
    try:
        pipe.enable_vae_tiling()
        logger.info("✅ 已启用VAE平铺优化")
    except Exception as e:
        logger.warning(f"VAE平铺优化失败: {e}")
    
    # 总结优化状态
    optimizations = []
    optimizations.append("✅ Float16精度")
    optimizations.append("✅ 注意力切片")
    if xformers_available:
        optimizations.append("✅ xFormers加速")
    else:
        optimizations.append("⚠️  xFormers不可用")
    optimizations.append("✅ VAE优化")
    optimizations.append("✅ 混合精度推理")
    
    logger.info(f"GPU优化状态: {' | '.join(optimizations)}")

logger.info("AI模型加载成功。")

# 加载IP-Adapter模型用于风格迁移
logger.info("加载IP-Adapter模型...")
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 使用更兼容的IP-Adapter加载方式
        pipe.load_ip_adapter(
            "h94/IP-Adapter", 
            subfolder="models", 
            weight_name="ip-adapter_sd15.bin",
            image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            # 添加兼容性参数
            low_cpu_mem_usage=True,
            torch_dtype=dtype
        )
    logger.info("IP-Adapter加载成功。")
except Exception as e:
    logger.warning(f"IP-Adapter加载失败: {e}")
    logger.info("将继续使用ControlNet进行生成，但无法使用风格迁移功能")
    # 设置标志表示IP-Adapter不可用
    pipe.ip_adapter_available = False
else:
    pipe.ip_adapter_available = True

# --- Helper Functions ---
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
COLOR_MAP = {
    "building": "#FF5733", "highway": "#C70039", "natural=water": "#33A8FF",
    "waterway": "#33A8FF", "landuse=forest": "#2ECC71", "leisure=park": "#27AE60",
    "barrier": "#9B59B6", "default": "#FDFEFE"
}

def get_bounding_box_details(coordinates: List[GeoCoordinate]):
    min_lat = min(c.lat for c in coordinates)
    max_lat = max(c.lat for c in coordinates)
    min_lng = min(c.lng for c in coordinates)
    max_lng = max(c.lng for c in coordinates)
    bbox_str = f"{min_lat},{min_lng},{max_lat},{max_lng}"
    return {"min_lat": min_lat, "max_lat": max_lat, "min_lng": min_lng, "max_lng": max_lng, "bbox_str": bbox_str}

def build_overpass_query(bbox_str: str):
    query = f'''[out:json][timeout:25];
    (
      way["building"]({bbox_str}); way["highway"]({bbox_str}); way["natural"="water"]({bbox_str});
      way["waterway"]({bbox_str}); way["landuse"="forest"]({bbox_str}); way["leisure"="park"]({bbox_str});
      way["barrier"]({bbox_str});
    );
    out body;>;out skel qt;'''
    return query

def create_semantic_map(osm_data, bbox_details, save_path):
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "black")
    draw = ImageDraw.Draw(img)
    nodes = {node['id']: node for node in osm_data['elements'] if node['type'] == 'node'}
    ways = [elem for elem in osm_data['elements'] if elem['type'] == 'way']
    lat_range = bbox_details["max_lat"] - bbox_details["min_lat"]
    lng_range = bbox_details["max_lng"] - bbox_details["min_lng"]
    if lat_range == 0 or lng_range == 0: return None
    
    def scale_coords(lon, lat):
        x = ((lon - bbox_details["min_lng"]) / lng_range) * IMAGE_WIDTH
        y = ((bbox_details["max_lat"] - lat) / lat_range) * IMAGE_HEIGHT
        return (x, y)
    
    for way in ways:
        points = [nodes.get(node_id) for node_id in way['nodes']]
        points = [p for p in points if p]
        if len(points) < 2: continue
        pixel_coords = [scale_coords(p['lon'], p['lat']) for p in points]
        color = COLOR_MAP["default"]
        for key, hex_color in COLOR_MAP.items():
            tag_key, tag_val = (key.split('=') + [None])[:2]
            if tag_key in way['tags'] and (tag_val is None or way['tags'][tag_key] == tag_val):
                color = hex_color
                break
        
        line_width = 5
        if way['nodes'][0] == way['nodes'][-1] and len(points) > 2:
            draw.polygon(pixel_coords, fill=color)
            draw.line(pixel_coords + [pixel_coords[0]], fill=color, width=line_width)
        else:
            draw.line(pixel_coords, fill=color, width=line_width)

    img.save(save_path)
    logger.info(f"语义地图保存到 {save_path}")
    return img

# --- Background Task ---
def run_generation_task(job_id: str, coordinates: List[GeoCoordinate]):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(DEBUG_DIR, f"{timestamp}_{job_id[:8]}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"[Job {job_id}] 开始任务 (GPU优化版本)。调试文件保存在: {run_dir}")
    
    try:
        bbox_details = get_bounding_box_details(coordinates)
        query = build_overpass_query(bbox_details["bbox_str"])
        logger.info(f"[Job {job_id}] 获取OSM数据...")
        response = requests.post(OVERPASS_URL, data=query)
        response.raise_for_status()
        osm_data = response.json()
        logger.info(f"[Job {job_id}] 找到 {len(osm_data.get('elements', []))} 个地图元素。")
        
        semantic_image = create_semantic_map(osm_data, bbox_details, os.path.join(run_dir, "semantic_map.png"))
        if semantic_image is None: 
            raise ValueError("创建语义地图失败，区域可能太小。")

        if semantic_image.getbbox() is None:
            logger.error(f"[Job {job_id}] 生成的语义地图完全是黑色。")
            raise ValueError("生成的语义地图为空白。")
        
        logger.info(f"[Job {job_id}] 语义地图创建成功。")

        logger.info(f"[Job {job_id}] 为Canny ControlNet预处理图像...")
        semantic_array = np.array(semantic_image)
        gray_image = cv2.cvtColor(semantic_array, cv2.COLOR_RGB2GRAY)
        canny_image = cv2.Canny(gray_image, 100, 200)
        control_image_canny = Image.fromarray(canny_image).convert("RGB")
        control_image_canny.save(os.path.join(run_dir, "control_canny.png"))
        
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

        # 根据IP-Adapter可用性调整提示词 - 优化版本，减少无中生有
        if use_ip_adapter:
            # 当使用IP-Adapter时，强调忠实于原始结构
            prompt = "A clean and accurate map, aerial view, simple style, only show existing roads, buildings, green areas and water bodies, no additional decorations, no text, no people, no vehicles, masterpiece, best quality"
            pipe.set_ip_adapter_scale(0.5)  # 降低风格影响，更忠实于结构
            logger.info(f"[Job {job_id}] 使用IP-Adapter + ControlNet模式（优化版）")
        else:
            # 不使用IP-Adapter时，同样强调结构忠实性
            prompt = "A clean hand-drawn style map, watercolor painting, aerial view, simple artistic style, only show existing roads, buildings, green areas and water bodies, no additional decorations, no text, no people, no vehicles, masterpiece, best quality"
            logger.info(f"[Job {job_id}] 使用基础ControlNet模式（优化版）")
        
        # 加强负面提示词，明确禁止添加不存在的元素
        negative_prompt = "blurry, low quality, distorted, text, words, letters, signs, labels, people, persons, humans, cars, vehicles, boats, ships, playground equipment, benches, furniture, decorations, icons, symbols, additional buildings, extra structures, fantasy elements, cartoon characters, ugly, boring, plain"
        
        logger.info(f"[Job {job_id}] 开始AI生成 ({'IP-Adapter + ControlNet' if use_ip_adapter else 'ControlNet'}, GPU优化加速)...")
        
        # GPU内存管理
        if device == "cuda":
            torch.cuda.empty_cache()
        
        generator = torch.Generator(device=device).manual_seed(1337)
        
        # GPU优化的推理参数 - 减少推理步数提升速度
        inference_steps = 30 if device == "cuda" else 8  # 从15步减少到10步
        
        # 使用推理模式和混合精度优化
        with torch.inference_mode():  # 更高效的推理模式
            if device == "cuda" and dtype == torch.float16:
                # 使用自动混合精度加速
                with torch.cuda.amp.autocast():
                    # 根据IP-Adapter可用性使用不同的生成参数
                    generation_params = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "image": control_image_canny,  # ControlNet的结构图
                        "num_inference_steps": inference_steps,
                        "controlnet_conditioning_scale": 1.2,  # 提高到1.2，更严格遵循结构
                        "guidance_scale": 6.0,  # 降低到6.0，减少过度生成
                        "generator": generator,
                        "output_type": "np"  # 直接输出numpy数组
                    }
                    
                    # 如果IP-Adapter可用，添加风格图片参数
                    if use_ip_adapter and style_image is not None:
                        generation_params["ip_adapter_image"] = style_image  # IP-Adapter的风格图
                    
                    output_images = pipe(**generation_params).images
            else:
                # CPU模式或float32模式
                generation_params = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": control_image_canny,  # ControlNet的结构图
                    "num_inference_steps": inference_steps,
                    "controlnet_conditioning_scale": 1.2,  # 提高到1.2，更严格遵循结构
                    "guidance_scale": 6.0,  # 降低到6.0，减少过度生成
                    "generator": generator,
                    "output_type": "np"  # 直接输出numpy数组
                }
                
                # 如果IP-Adapter可用，添加风格图片参数
                if use_ip_adapter and style_image is not None:
                    generation_params["ip_adapter_image"] = style_image  # IP-Adapter的风格图
                
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
    if device == "cuda":
        return {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            "gpu_memory_cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        }
    else:
        return {"device": device, "message": "GPU not available"}

# --- Main Execution (for direct run) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("直接运行GPU优化版本脚本，启动uvicorn服务...")
    uvicorn.run(app, host="127.0.0.1", port=8003)