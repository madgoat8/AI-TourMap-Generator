from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict
import requests
from PIL import Image, ImageDraw
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import cv2
import os
import uuid
import base64
from io import BytesIO
import webbrowser
import logging
import datetime

# --- Configuration & Global State ---
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    filename='generation.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Create a directory for debug runs
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

# --- Startup Event ---
@app.on_event("startup")
async def on_startup():
    global HAS_OPENED_BROWSER
    if not HAS_OPENED_BROWSER:
        try:
            url = "http://127.0.0.1:8000"
            logger.info(f"Server started. Opening browser to: {url}")
            webbrowser.open(url)
            HAS_OPENED_BROWSER = True
        except Exception as e:
            logger.error(f"Could not open browser: {e}")

# --- AI Model Loading ---
logger.info("Checking for available hardware accelerators...")
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
logger.info(f"Using device: {device}")

logger.info("Loading ControlNet model...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", 
    torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
    cache_dir=cache_dir
)

logger.info("Loading Stable Diffusion pipeline...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
    cache_dir=cache_dir,
    safety_checker=None
)

if device in ["cuda", "mps"]:
    pipe.to(device)

pipe.enable_attention_slicing()
logger.info("AI Models loaded successfully.")

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
        
        # Draw thicker lines and outlines
        line_width = 5
        if way['nodes'][0] == way['nodes'][-1] and len(points) > 2: # It's a polygon
            draw.polygon(pixel_coords, fill=color) # Fill the shape
            # Draw a thick outline over the filled shape
            draw.line(pixel_coords + [pixel_coords[0]], fill=color, width=line_width)
        else: # It's a line
            draw.line(pixel_coords, fill=color, width=line_width)

    img.save(save_path)
    logger.info(f"Semantic map saved to {save_path}")
    return img

def build_poi_query(bbox_str: str):
    query = f'''[out:json][timeout:25];
    (
      node["tourism"]({bbox_str}); node["amenity"]({bbox_str}); node["historic"]({bbox_str}); node["shop"]({bbox_str});
      way["tourism"]({bbox_str}); way["amenity"]({bbox_str}); way["historic"]({bbox_str}); way["shop"]({bbox_str});
    );
    out center;'''
    return query

# --- Background Task ---
def run_generation_task(job_id: str, coordinates: List[GeoCoordinate]):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(DEBUG_DIR, f"{timestamp}_{job_id[:8]}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"[Job {job_id}] Starting task. Debug files will be saved in: {run_dir}")
    try:
        bbox_details = get_bounding_box_details(coordinates)
        query = build_overpass_query(bbox_details["bbox_str"])
        logger.info(f"[Job {job_id}] Fetching OSM data for basemap...")
        response = requests.post(OVERPASS_URL, data=query)
        response.raise_for_status()
        osm_data = response.json()
        logger.info(f"[Job {job_id}] Found {len(osm_data.get('elements', []))} basemap elements.")
        
        semantic_image = create_semantic_map(osm_data, bbox_details, os.path.join(run_dir, "semantic_map.png"))
        if semantic_image is None: raise ValueError("Failed to create semantic map, area might be too small.")

        # Diagnostic check for blank semantic map
        if semantic_image.getbbox() is None:
            logger.error(f"[Job {job_id}] The generated semantic_map.png is entirely black. No features were drawn.")
            raise ValueError("The generated semantic map was blank. Check GIS data or drawing logic.")
        else:
            logger.info(f"[Job {job_id}] Semantic map created successfully.")

        # --- Canny Edge Preprocessing ---
        logger.info(f"[Job {job_id}] Preprocessing image for Canny ControlNet...")
        # Convert to numpy array for OpenCV processing
        semantic_array = np.array(semantic_image)
        # Convert to grayscale
        gray_image = cv2.cvtColor(semantic_array, cv2.COLOR_RGB2GRAY)
        # Apply Canny edge detection with optimized parameters
        canny_image = cv2.Canny(gray_image, 100, 200)
        # Convert back to PIL Image and then to RGB for the pipeline
        control_image_canny = Image.fromarray(canny_image).convert("RGB")
        control_image_canny.save(os.path.join(run_dir, "control_canny.png"))
        
        # Optimized prompt for better results
        prompt = "hand drawn map illustration, watercolor style, aerial view, colorful buildings and parks, roads and pathways, artistic map design, detailed topographic illustration"
        negative_prompt = "blurry, dark, black, monochrome, low quality, distorted"
        
        # Optimized ControlNet parameters for better generation
        # Optimized ControlNet parameters with error handling
        try:
            logger.info(f"[Job {job_id}] Starting AI generation with optimized parameters...")
            
            # For MPS compatibility, use lower precision and add memory management
            with torch.no_grad():
                if device == "mps":
                    # MPS-specific optimizations
                    torch.mps.empty_cache()
                    output = pipe(
                        prompt, 
                        negative_prompt=negative_prompt,
                        image=control_image_canny, 
                        num_inference_steps=25,
                        controlnet_conditioning_scale=0.6,
                        guidance_scale=10.0,
                        generator=torch.Generator(device=device).manual_seed(42)
                    )
                else:
                    output = pipe(
                        prompt, 
                        negative_prompt=negative_prompt,
                        image=control_image_canny, 
                        num_inference_steps=30,
                        controlnet_conditioning_scale=0.7,
                        guidance_scale=12.0,
                        strength=0.8
                    )
            
            logger.info(f"[Job {job_id}] AI generation completed successfully.")
            logger.info(f"[Job {job_id}] AI generation completed successfully.")
            
            # 检查和修复生成图像的数值问题
            final_image = output.images[0]
            final_array = np.array(final_image)
            
            # 检查是否包含无效值
            if np.any(np.isnan(final_array)) or np.any(np.isinf(final_array)):
                logger.warning(f"[Job {job_id}] Generated image contains invalid values, attempting to fix...")
                # 将无效值替换为0
                final_array = np.nan_to_num(final_array, nan=0.0, posinf=255.0, neginf=0.0)
                final_image = Image.fromarray(final_array.astype(np.uint8))
            
            # 检查是否为纯黑图像
            if np.all(final_array < 10):
                logger.warning(f"[Job {job_id}] Generated image is nearly black, trying alternative approach...")
                # 尝试使用CPU模式重新生成
                if device == "mps":
                    logger.info(f"[Job {job_id}] Switching to CPU mode for fallback generation...")
                    pipe_cpu = StableDiffusionControlNetPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        controlnet=controlnet,
                        torch_dtype=torch.float32,
                        cache_dir=cache_dir,
                        safety_checker=None
                    )
                    
                    cpu_output = pipe_cpu(
                        "colorful hand drawn map, vibrant illustration", 
                        image=control_image_canny, 
                        num_inference_steps=15,
                        controlnet_conditioning_scale=0.8,
                        guidance_scale=8.0
                    )
                    final_image = cpu_output.images[0]
                    logger.info(f"[Job {job_id}] CPU fallback generation completed.")
            
        except Exception as gen_error:
            logger.error(f"[Job {job_id}] AI generation failed: {gen_error}")
            # Fallback: try with minimal parameters
            logger.info(f"[Job {job_id}] Attempting fallback generation...")
            try:
                if device == "mps":
                    torch.mps.empty_cache()
                output = pipe(
                    "simple hand drawn map, colorful illustration", 
                    image=control_image_canny, 
                    num_inference_steps=15,
                    controlnet_conditioning_scale=0.5,
                    guidance_scale=7.5
                )
                logger.info(f"[Job {job_id}] Fallback generation succeeded.")
            except Exception as fallback_error:
                logger.error(f"[Job {job_id}] Fallback generation also failed: {fallback_error}")
                raise ValueError(f"AI generation failed: {gen_error}")
        final_image = output.images[0]
        final_image.save(os.path.join(run_dir, "final_map.png"))

        buffered = BytesIO()
        final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info(f"[Job {job_id}] Task complete.")
        JOBS[job_id] = {
            "status": "complete",
            "result": {
                "image_base64": img_str,
                "bounds": [[bbox_details["min_lat"], bbox_details["min_lng"]], [bbox_details["max_lat"], bbox_details["max_lng"]]]
            }}
    except Exception as e:
        logger.error(f"[Job {job_id}] Task failed: {e}", exc_info=True)
        JOBS[job_id] = {"status": "failed", "error": str(e)}

# --- API Endpoints ---
@app.get("/", response_class=FileResponse)
async def read_index():
    return "index.html"

@app.post("/api/extract_pois")
def extract_pois(area: AreaSelection):
    bbox_details = get_bounding_box_details(area.coordinates)
    query = build_poi_query(bbox_details["bbox_str"])
    logger.info("Fetching POIs from Overpass API...")
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
        logger.info(f"Found {len(pois)} POIs.")
        return pois
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching POI data: {e}")
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