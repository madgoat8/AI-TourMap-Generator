"""
AI模型加载器
"""
import torch
import warnings
import logging
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from config import ModelConfig, GPUConfig, CACHE_DIR

logger = logging.getLogger(__name__)

class ModelLoader:
    """AI模型加载器"""
    
    def __init__(self):
        self.device = None
        self.dtype = None
        self.controlnet = None
        self.pipe = None
        self._setup_device()
    
    def _setup_device(self):
        """设置计算设备和数据类型"""
        if torch.cuda.is_available():
            self.device = "cuda"
            # GPU内存优化
            torch.cuda.empty_cache()
            # 启用优化
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info(f"使用GPU设备: {torch.cuda.get_device_name()}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = "cpu"
            logger.warning("GPU不可用，使用CPU运行")
        
        # 设置数据类型
        self.dtype = torch.float16 if (self.device == "cuda" and GPUConfig.USE_FLOAT16) else torch.float32
        logger.info(f"使用设备: {self.device}, 数据类型: {self.dtype}")
    
    def load_controlnet(self) -> ControlNetModel:
        """
        加载ControlNet模型
        
        Returns:
            ControlNet模型实例
        """
        logger.info(f"加载ControlNet模型: {ModelConfig.CONTROLNET_MODEL}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.controlnet = ControlNetModel.from_pretrained(
                ModelConfig.CONTROLNET_MODEL,
                torch_dtype=self.dtype,
                cache_dir=CACHE_DIR
            )
        
        logger.info("ControlNet模型加载成功")
        return self.controlnet
    
    def load_stable_diffusion_pipeline(self) -> StableDiffusionControlNetPipeline:
        """
        加载Stable Diffusion管道
        
        Returns:
            Stable Diffusion管道实例
        """
        if self.controlnet is None:
            self.load_controlnet()
        
        logger.info(f"加载Stable Diffusion管道: {ModelConfig.SD_MODEL}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                ModelConfig.SD_MODEL,
                controlnet=self.controlnet,
                torch_dtype=self.dtype,
                cache_dir=CACHE_DIR,
                safety_checker=None
            ).to(self.device)
        
        logger.info("Stable Diffusion管道加载成功")
        return self.pipe
    
    def setup_gpu_optimizations(self):
        """设置GPU优化"""
        if self.device != "cuda" or self.pipe is None:
            return
        
        optimizations = []
        
        # 启用内存高效注意力机制
        if GPUConfig.ENABLE_ATTENTION_SLICING:
            self.pipe.enable_attention_slicing()
            optimizations.append("✅ 注意力切片")
            logger.info("✅ 已启用注意力切片优化")
        
        # 尝试启用xFormers优化
        xformers_available = False
        if GPUConfig.ENABLE_XFORMERS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import xformers
                    self.pipe.enable_xformers_memory_efficient_attention()
                    xformers_available = True
                    optimizations.append("✅ xFormers加速")
                    logger.info("✅ 已启用xFormers内存高效注意力机制")
            except (ImportError, ModuleNotFoundError):
                optimizations.append("⚠️  xFormers未安装")
                logger.info("ℹ️  xFormers未安装，使用标准注意力机制")
            except Exception:
                optimizations.append("⚠️  xFormers不可用")
                logger.info("ℹ️  xFormers在当前环境不可用，使用标准注意力机制")
        
        # 启用VAE切片以节省内存
        if GPUConfig.ENABLE_VAE_SLICING:
            try:
                self.pipe.enable_vae_slicing()
                optimizations.append("✅ VAE切片")
                logger.info("✅ 已启用VAE切片优化")
            except Exception as e:
                logger.warning(f"VAE切片优化失败: {e}")
        
        # 启用VAE平铺（对大图像有效）
        if GPUConfig.ENABLE_VAE_TILING:
            try:
                self.pipe.enable_vae_tiling()
                optimizations.append("✅ VAE平铺")
                logger.info("✅ 已启用VAE平铺优化")
            except Exception as e:
                logger.warning(f"VAE平铺优化失败: {e}")
        
        # 添加基础优化信息
        optimizations.insert(0, "✅ Float16精度" if self.dtype == torch.float16 else "✅ Float32精度")
        optimizations.append("✅ 混合精度推理")
        
        logger.info(f"GPU优化状态: {' | '.join(optimizations)}")
    
    def load_ip_adapter(self):
        """加载IP-Adapter模型"""
        if self.pipe is None:
            raise RuntimeError("必须先加载Stable Diffusion管道")
        
        logger.info("加载IP-Adapter模型...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.pipe.load_ip_adapter(
                    ModelConfig.IP_ADAPTER_MODEL,
                    subfolder="models",
                    weight_name=ModelConfig.IP_ADAPTER_WEIGHT,
                    image_encoder_path=ModelConfig.IMAGE_ENCODER,
                    low_cpu_mem_usage=True,
                    torch_dtype=self.dtype
                )
            self.pipe.ip_adapter_available = True
            logger.info("IP-Adapter加载成功")
        except Exception as e:
            logger.warning(f"IP-Adapter加载失败: {e}")
            logger.info("将继续使用ControlNet进行生成，但无法使用风格迁移功能")
            self.pipe.ip_adapter_available = False
    
    def get_device_info(self) -> dict:
        """获取设备信息"""
        if self.device == "cuda":
            return {
                "device": self.device,
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                "gpu_memory_cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
                "dtype": str(self.dtype)
            }
        else:
            return {
                "device": self.device,
                "message": "GPU not available",
                "dtype": str(self.dtype)
            }