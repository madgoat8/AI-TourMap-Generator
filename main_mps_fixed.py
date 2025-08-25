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

# --- Configuration & Global State ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    filename='generation.log',
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

# --- FastAPI App ---
app = FastAPI()

@app.on_event("startup")
async def on_startup():
    global HAS_OPENED_BROWSER
    if not HAS_OPENED_BROWSER:
        try:
            url = "http://127.0.0.1:8002"
            logger.info(f"MPS修复版本服务启动。打开浏览器: {url}")
            webbrowser.open(url)
            HAS_OPENED_BROWSER = True
        except Exception as e:
            logger.error(f"无法打开浏览器: {e}")

# --- AI Model Loading with MPS Optimizations ---
logger.info("MPS设备优化加载...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"使用设备: {device}")

# MPS优化设置
if device == "mps":
    # 设置MPS内存分配策略
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    logger.info("应用MPS内存优化设置")

logger.info("加载ControlNet模型...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", 
    torch_dtype=torch.float32,  # 使用float32提高数值稳定性
    cache_dir=cache_dir
)

logger.info("加载Stable Diffusion管道...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float32,  # 使用float32提高数值稳定性
    cache_dir=cache_dir,
    safety_checker=None
)

if device == "mps":
    pipe.to(device)
    # 应用MPS优化
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    logger.info("应用MPS设备优化")

logger.info("MPS优化版本AI模型加载成功。")

# --- Helper Functions ---
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 768
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

def build_poi_query(bbox_str: str):
    query = f'''[out:json][timeout:25];
    (
      node["tourism"]({bbox_str}); node["amenity"]({bbox_str}); node["historic"]({bbox_str}); node["shop"]({bbox_str});
      way["tourism"]({bbox_str}); way["amenity"]({bbox_str}); way["historic"]({bbox_str}); way["shop"]({bbox_str});
    );
    out center;'''
    return query

def fix_image_values(image_array):
    """修复图像中的无效数值"""
    # 检查并修复NaN和无穷大值
    if np.any(np.isnan(image_array)) or np.any(np.isinf(image_array)):
        logger.warning("检测到无效像素值，正在修复...")
        image_array = np.nan_to_num(image_array, nan=128.0, posinf=255.0, neginf=0.0)
    
    # 确保值在有效范围内
    image_array = np.clip(image_array, 0, 255)
    
    return image_array.astype(np.uint8)

# --- Background Task ---
def run_generation_task(job_id: str, coordinates: List[GeoCoordinate]):
    run_dir = os.path.join(DEBUG_DIR, job_id)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"[Job {job_id}] 开始任务 (MPS优化版本)。调试文件保存在: {run_dir}")
    
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
        else:
            logger.info(f"[Job {job_id}] 语义地图创建成功。")

        # --- Canny边缘预处理 ---
        logger.info(f"[Job {job_id}] 为Canny ControlNet预处理图像...")
        semantic_array = np.array(semantic_image)
        gray_image = cv2.cvtColor(semantic_array, cv2.COLOR_RGB2GRAY)
        canny_image = cv2.Canny(gray_image, 100, 200)
        control_image_canny = Image.fromarray(canny_image).convert("RGB")
        control_image_canny.save(os.path.join(run_dir, "control_canny.png"))
        
        # MPS优化的生成参数
        prompt = "vibrant colorful building illustration, bright red walls, blue sky, detailed architecture, high contrast, watercolor style map"
        negative_prompt = "black, dark, monochrome, low contrast, blurry, noise, artifacts, low quality"
        
        logger.info(f"[Job {job_id}] 开始AI生成 (MPS优化模式)...")
        
        # MPS优化生成
        if device == "mps":
            torch.mps.empty_cache()
        
        # 使用固定种子确保可重现性
        generator = torch.Generator(device=device).manual_seed(42)
        
        with torch.no_grad():
            # 使用测试验证的成功参数 (float32精度配置)
            output = pipe(
                prompt=prompt,
                image=control_image_canny,
                num_inference_steps=15,  # 测试成功的步数
                controlnet_conditioning_scale=0.7,  # 测试成功的控制强度
                guidance_scale=7.5,  # 测试成功的引导强度
                generator=generator
            )
        
        # 获取生成的图像并进行数值修复
        raw_image = output.images[0]
        raw_array = np.array(raw_image)
        
        # 应用数值修复
        fixed_array = fix_image_values(raw_array)
        final_image = Image.fromarray(fixed_array)
        
        final_image.save(os.path.join(run_dir, "final_map.png"))
        
        # 验证生成结果
        if np.all(fixed_array < 10):
            logger.warning(f"[Job {job_id}] 生成的图像仍然很暗，尝试备用方法...")
            
            # 备用生成：更强的颜色提示
            with torch.no_grad():
                if device == "mps":
                    torch.mps.empty_cache()
                
                backup_output = pipe(
                    prompt="bright colorful house, vivid red building, blue windows, green trees, yellow sun, high saturation, cartoon style", 
                    negative_prompt="black, dark, gray, monochrome, low saturation",
                    image=control_image_canny, 
                    num_inference_steps=30,
                    controlnet_conditioning_scale=0.5,
                    guidance_scale=12.0,
                    generator=torch.Generator(device=device).manual_seed(123)
                )
            
            backup_array = fix_image_values(np.array(backup_output.images[0]))
            final_image = Image.fromarray(backup_array)
            final_image.save(os.path.join(run_dir, "final_map.png"))
            logger.info(f"[Job {job_id}] 使用备用方法重新生成")
        
        final_array = np.array(final_image)
        logger.info(f"[Job {job_id}] 生成完成！像素值范围: {final_array.min()} - {final_array.max()}")

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

# --- API Endpoints ---
@app.get("/", response_class=FileResponse)
async def read_index():
    return "index.html"

@app.post("/api/extract_pois")
def extract_pois(area: AreaSelection):
    bbox_details = get_bounding_box_details(area.coordinates)
    query = build_poi_query(bbox_details["bbox_str"])
    logger.info("从Overpass API获取POIs...")
    try:
        response = requests.post(OVERPASS_URL, data=query)
        response.raise_for_status()
        osm_data = response.json()
        pois = []
        for elem in osm_data.get('elements', []):
            if 'tags' in elem and 'name' in elem['tags']:
                lat = elem.get('lat') or elem.get('center', {}).get('lat')
                lon = elem.get('lon') or elem.get('center', {}).get('lon')
                if lat and lon:
                    pois.append({"name": elem['tags']['name'], "lat": lat, "lon": lon})
        logger.info(f"找到 {len(pois)} 个POIs。")
        return pois
    except requests.exceptions.RequestException as e:
        logger.error(f"获取POI数据错误: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/start_generation")
def start_generation(area: AreaSelection, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "processing"}
    background_tasks.add_task(run_generation_task, job_id, area.coordinates)
    return {"job_id": job_id}

@app.get("/api/job_status/{job_id}")
def get_job_status(job_id: str):
    return JOBS.get(job_id, {"status": "not_found"})