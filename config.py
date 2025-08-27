"""
配置文件 - 集中管理所有配置项
"""
import os

# === 目录配置 ===
DEBUG_DIR = "debug_runs"
CACHE_DIR = "./model_cache"

# === 图像配置 ===
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# === API配置 ===
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8003

# === 颜色映射 ===
COLOR_MAP = {
    "building": "#FF5733", 
    "highway": "#C70039", 
    "natural=water": "#33A8FF",
    "waterway": "#33A8FF", 
    "landuse=forest": "#2ECC71", 
    "leisure=park": "#27AE60",
    "barrier": "#9B59B6", 
    "default": "#FDFEFE"
}

# === AI模型配置 ===
class ModelConfig:
    # 模型路径
    CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
    SD_MODEL = "runwayml/stable-diffusion-v1-5"
    IP_ADAPTER_MODEL = "h94/IP-Adapter"
    IP_ADAPTER_WEIGHT = "ip-adapter_sd15.bin"
    IMAGE_ENCODER = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    
    # 生成参数
    INFERENCE_STEPS_GPU = 30
    INFERENCE_STEPS_CPU = 8
    CONTROLNET_SCALE = 1.2
    GUIDANCE_SCALE = 6.0
    IP_ADAPTER_SCALE = 0.5
    RANDOM_SEED = 1337
    
    # Canny参数
    CANNY_LOW_THRESHOLD = 100
    CANNY_HIGH_THRESHOLD = 200
    LINE_WIDTH = 5

# === 提示词配置 ===
class PromptConfig:
    # 正面提示词
    PROMPT_WITH_IP_ADAPTER = "A clean and accurate map, aerial view, simple style, only show existing roads, buildings, green areas and water bodies, no additional decorations, no text, no people, no vehicles, masterpiece, best quality"
    
    PROMPT_WITHOUT_IP_ADAPTER = "A clean hand-drawn style map, watercolor painting, aerial view, simple artistic style, only show existing roads, buildings, green areas and water bodies, no additional decorations, no text, no people, no vehicles, masterpiece, best quality"
    
    # 负面提示词
    NEGATIVE_PROMPT = "blurry, low quality, distorted, text, words, letters, signs, labels, people, persons, humans, cars, vehicles, boats, ships, playground equipment, benches, furniture, decorations, icons, symbols, additional buildings, extra structures, fantasy elements, cartoon characters, ugly, boring, plain"

# === GPU优化配置 ===
class GPUConfig:
    # 环境变量
    XFORMERS_FORCE_DISABLE_TRITON = "1"
    PYTHONWARNINGS = "ignore"
    
    # 优化选项
    ENABLE_ATTENTION_SLICING = True
    ENABLE_VAE_SLICING = True
    ENABLE_VAE_TILING = True
    ENABLE_XFORMERS = True
    USE_FLOAT16 = True

# === 日志配置 ===
class LogConfig:
    LEVEL = "INFO"
    FORMAT = '%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
    FILENAME = 'generation_gpu.log'
    FILEMODE = 'a'

# === 初始化函数 ===
def ensure_directories():
    """确保必要的目录存在"""
    for directory in [DEBUG_DIR, CACHE_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

def setup_environment():
    """设置环境变量"""
    os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = GPUConfig.XFORMERS_FORCE_DISABLE_TRITON
    os.environ["PYTHONWARNINGS"] = GPUConfig.PYTHONWARNINGS