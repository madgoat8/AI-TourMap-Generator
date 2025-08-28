"""
天地图卫星图像下载器
支持根据经纬度坐标下载高分辨率卫星图像
"""
import os
import math
import asyncio
import aiohttp
import logging
from PIL import Image
from typing import Tuple, List
import random
import time
from config import TianDiTuConfig

class TianDiTuDownloader:
    def __init__(self, api_key: str = None):
        """
        初始化天地图下载器
        
        Args:
            api_key: 天地图API密钥，如果不提供则使用配置文件中的密钥
        """
        self.api_key = api_key or TianDiTuConfig.API_KEY
        self.tile_size = TianDiTuConfig.TILE_SIZE
        self.max_zoom = TianDiTuConfig.MAX_ZOOM_LEVEL
        self.min_zoom = TianDiTuConfig.MIN_ZOOM_LEVEL
        self.max_retries = TianDiTuConfig.MAX_RETRIES
        self.timeout = TianDiTuConfig.TIMEOUT
        self.max_concurrent = TianDiTuConfig.MAX_CONCURRENT_DOWNLOADS
        self.delay = TianDiTuConfig.DELAY_BETWEEN_REQUESTS
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def deg2num(self, lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """
        将经纬度转换为瓦片坐标
        
        Args:
            lat_deg: 纬度
            lon_deg: 经度
            zoom: 缩放级别
            
        Returns:
            (x, y): 瓦片坐标
        """
        # 验证输入范围
        if not (-90 <= lat_deg <= 90):
            raise ValueError(f"纬度超出范围: {lat_deg} (应在-90到90之间)")
        if not (-180 <= lon_deg <= 180):
            raise ValueError(f"经度超出范围: {lon_deg} (应在-180到180之间)")
        if not (0 <= zoom <= 20):
            raise ValueError(f"缩放级别超出范围: {zoom} (应在0到20之间)")
        
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        
        # 计算X坐标 (经度)
        x = (lon_deg + 180.0) / 360.0 * n
        x_int = int(x)
        
        # 计算Y坐标 (纬度) - 使用Web墨卡托投影
        y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        y_int = int(y)
        
        # 确保坐标在有效范围内
        max_tile = int(n) - 1
        x_int = max(0, min(max_tile, x_int))
        y_int = max(0, min(max_tile, y_int))
        
        self.logger.debug(f"坐标转换: ({lat_deg:.6f}, {lon_deg:.6f}) -> 瓦片({x_int}, {y_int}) at zoom {zoom}")
        
        return (x_int, y_int)
    
    def num2deg(self, x: int, y: int, zoom: int) -> Tuple[float, float]:
        """
        将瓦片坐标转换为经纬度
        
        Args:
            x: 瓦片X坐标
            y: 瓦片Y坐标
            zoom: 缩放级别
            
        Returns:
            (lat, lon): 经纬度
        """
        n = 2.0 ** zoom
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)
    
    def calculate_tiles_for_bbox(self, north: float, south: float, east: float, west: float, zoom: int) -> List[Tuple[int, int]]:
        """
        计算边界框内所有需要的瓦片坐标
        
        Args:
            north, south, east, west: 边界框坐标
            zoom: 缩放级别
            
        Returns:
            瓦片坐标列表
        """
        self.logger.debug(f"计算瓦片范围: 北={north}, 南={south}, 东={east}, 西={west}, 缩放={zoom}")
        
        # 验证边界框
        if north <= south:
            raise ValueError(f"北纬({north})必须大于南纬({south})")
        if east <= west:
            raise ValueError(f"东经({east})必须大于西经({west})")
        
        # 转换边界框的四个角点到瓦片坐标
        # 西北角
        x_min, y_min = self.deg2num(north, west, zoom)
        # 东南角  
        x_max, y_max = self.deg2num(south, east, zoom)
        
        self.logger.debug(f"瓦片坐标范围: x({x_min}-{x_max}), y({y_min}-{y_max})")
        
        # 确保坐标顺序正确
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
            
        self.logger.debug(f"修正后瓦片坐标范围: x({x_min}-{x_max}), y({y_min}-{y_max})")
        
        # 生成所有瓦片坐标
        tiles = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tiles.append((x, y))
        
        self.logger.info(f"计算得到 {len(tiles)} 个瓦片: x范围({x_min}-{x_max}), y范围({y_min}-{y_max})")
        
        if len(tiles) == 0:
            self.logger.error(f"计算出0个瓦片！边界框可能有问题")
            self.logger.error(f"输入: 北={north}, 南={south}, 东={east}, 西={west}")
            self.logger.error(f"瓦片坐标: 西北({x_min},{y_min}), 东南({x_max},{y_max})")
        
        return tiles
    
    def get_tile_url(self, x: int, y: int, zoom: int, layer_type: str = "satellite", use_backup: bool = False) -> str:
        """
        生成瓦片URL
        
        Args:
            x, y: 瓦片坐标
            zoom: 缩放级别
            layer_type: 图层类型 ("satellite" 或 "annotation")
            use_backup: 是否使用备用URL格式
            
        Returns:
            瓦片URL
        """
        server = random.choice(TianDiTuConfig.SERVERS)
        
        if use_backup:
            # 使用备用DataServer格式
            if layer_type == "satellite":
                url_template = TianDiTuConfig.SATELLITE_URL_BACKUP
            else:
                url_template = TianDiTuConfig.ANNOTATION_URL_BACKUP
        else:
            # 使用WMTS格式
            if layer_type == "satellite":
                url_template = TianDiTuConfig.SATELLITE_URL
            else:
                url_template = TianDiTuConfig.ANNOTATION_URL
            
        return url_template.format(
            s=server,
            x=x,
            y=y,
            z=zoom,
            api_key=self.api_key
        )
    
    async def download_tile(self, session: aiohttp.ClientSession, x: int, y: int, zoom: int, layer_type: str = "satellite") -> Image.Image:
        """
        下载单个瓦片
        
        Args:
            session: aiohttp会话
            x, y: 瓦片坐标
            zoom: 缩放级别
            layer_type: 图层类型
            
        Returns:
            PIL图像对象
        """
        # 尝试两种URL格式
        url_formats = [False, True]  # False=WMTS格式, True=备用DataServer格式
        
        for use_backup in url_formats:
            url = self.get_tile_url(x, y, zoom, layer_type, use_backup)
            format_name = "DataServer" if use_backup else "WMTS"
            self.logger.debug(f"下载瓦片 ({x}, {y}) 使用{format_name}格式: {url}")
            
            for attempt in range(self.max_retries):
                try:
                    await asyncio.sleep(self.delay)  # 添加延迟避免请求过快
                    
                    async with session.get(url, timeout=self.timeout) as response:
                        self.logger.debug(f"瓦片 ({x}, {y}) {format_name}格式 响应状态: {response.status}")
                        
                        if response.status == 200:
                            image_data = await response.read()
                            self.logger.debug(f"瓦片 ({x}, {y}) 数据大小: {len(image_data)} bytes")
                            
                            if len(image_data) > 0:
                                try:
                                    image = Image.open(io.BytesIO(image_data))
                                    # 验证图像是否有效
                                    image.verify()
                                    # 重新打开图像用于实际使用
                                    image = Image.open(io.BytesIO(image_data))
                                    self.logger.info(f"瓦片 ({x}, {y}) 使用{format_name}格式下载成功，尺寸: {image.size}")
                                    return image
                                except Exception as img_error:
                                    self.logger.warning(f"瓦片 ({x}, {y}) 图像解析失败: {img_error}")
                            else:
                                self.logger.warning(f"瓦片 ({x}, {y}) 返回空数据")
                        elif response.status == 403:
                            self.logger.warning(f"瓦片 ({x}, {y}) {format_name}格式 HTTP 403 (权限拒绝)")
                            break  # 403错误不重试，直接尝试下一种格式
                        else:
                            self.logger.warning(f"瓦片 ({x}, {y}) {format_name}格式 HTTP错误 {response.status}, 尝试 {attempt + 1}/{self.max_retries}")
                            
                except Exception as e:
                    self.logger.warning(f"瓦片 ({x}, {y}) {format_name}格式 下载异常，尝试 {attempt + 1}/{self.max_retries}: {e}")
                    
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.debug(f"瓦片 ({x}, {y}) 等待 {wait_time}秒后重试")
                    await asyncio.sleep(wait_time)  # 指数退避
            
            # 如果当前格式失败，尝试下一种格式
            self.logger.warning(f"瓦片 ({x}, {y}) {format_name}格式下载失败，尝试其他格式")
        
        # 如果所有格式都失败，返回空白瓦片
        self.logger.error(f"瓦片 ({x}, {y}) 所有格式下载失败，使用空白瓦片替代")
        return Image.new('RGB', (self.tile_size, self.tile_size), color=(200, 200, 200))  # 浅灰色便于识别
    
    async def download_tiles_batch(self, tiles: List[Tuple[int, int]], zoom: int, layer_type: str = "satellite") -> dict:
        """
        批量下载瓦片
        
        Args:
            tiles: 瓦片坐标列表
            zoom: 缩放级别
            layer_type: 图层类型
            
        Returns:
            瓦片字典 {(x, y): PIL.Image}
        """
        self.logger.info(f"开始批量下载 {len(tiles)} 个{layer_type}瓦片，缩放级别: {zoom}")
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def download_with_semaphore(session, x, y):
            async with semaphore:
                return await self.download_tile(session, x, y, zoom, layer_type)
        
        # 设置连接器参数
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        # 设置超时
        timeout = aiohttp.ClientTimeout(total=self.timeout * 2, connect=self.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            self.logger.info(f"创建HTTP会话，最大并发: {self.max_concurrent}")
            
            tasks = [download_with_semaphore(session, x, y) for x, y in tiles]
            
            # 分批处理以避免内存问题
            batch_size = 50
            all_images = []
            
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_tiles = tiles[i:i + batch_size]
                
                self.logger.info(f"处理批次 {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}, 瓦片数: {len(batch_tasks)}")
                
                try:
                    batch_images = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # 检查结果
                    valid_count = 0
                    for j, img in enumerate(batch_images):
                        if isinstance(img, Exception):
                            self.logger.error(f"瓦片 {batch_tiles[j]} 下载异常: {img}")
                            batch_images[j] = Image.new('RGB', (self.tile_size, self.tile_size), color=(200, 200, 200))
                        elif img and hasattr(img, 'size'):
                            valid_count += 1
                    
                    self.logger.info(f"批次完成: {valid_count}/{len(batch_tasks)} 个瓦片有效")
                    all_images.extend(batch_images)
                    
                except Exception as e:
                    self.logger.error(f"批次处理失败: {e}")
                    # 为失败的批次创建空白图像
                    all_images.extend([Image.new('RGB', (self.tile_size, self.tile_size), color=(200, 200, 200)) for _ in batch_tasks])
        
        result = {tiles[i]: all_images[i] for i in range(len(tiles))}
        
        # 统计下载结果
        valid_tiles = sum(1 for img in all_images if img and hasattr(img, 'size') and img.size[0] > 0)
        self.logger.info(f"批量下载完成: {valid_tiles}/{len(tiles)} 个瓦片有效")
        
        return result
    
    def stitch_tiles(self, tile_images: dict, tiles: List[Tuple[int, int]]) -> Image.Image:
        """
        拼接瓦片为完整图像
        
        Args:
            tile_images: 瓦片图像字典
            tiles: 瓦片坐标列表
            
        Returns:
            拼接后的完整图像
        """
        if not tiles:
            self.logger.warning("没有瓦片需要拼接")
            return Image.new('RGB', (self.tile_size, self.tile_size), color='white')
        
        # 计算拼接后图像的尺寸
        x_coords = [x for x, y in tiles]
        y_coords = [y for x, y in tiles]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        width = (max_x - min_x + 1) * self.tile_size
        height = (max_y - min_y + 1) * self.tile_size
        
        self.logger.info(f"拼接图像尺寸: {width}x{height}, 瓦片范围: x({min_x}-{max_x}), y({min_y}-{max_y})")
        
        # 创建拼接图像
        stitched = Image.new('RGB', (width, height), color='white')
        
        # 统计成功拼接的瓦片数量
        successful_tiles = 0
        failed_tiles = 0
        
        # 拼接每个瓦片
        for (x, y), image in tile_images.items():
            if image and hasattr(image, 'size') and image.size[0] > 0 and image.size[1] > 0:
                try:
                    paste_x = (x - min_x) * self.tile_size
                    paste_y = (y - min_y) * self.tile_size
                    
                    # 确保图像尺寸正确
                    if image.size != (self.tile_size, self.tile_size):
                        image = image.resize((self.tile_size, self.tile_size), Image.Resampling.LANCZOS)
                    
                    stitched.paste(image, (paste_x, paste_y))
                    successful_tiles += 1
                    self.logger.debug(f"瓦片 ({x}, {y}) 拼接成功，位置: ({paste_x}, {paste_y})")
                except Exception as e:
                    self.logger.warning(f"瓦片 ({x}, {y}) 拼接失败: {e}")
                    failed_tiles += 1
            else:
                self.logger.warning(f"瓦片 ({x}, {y}) 无效或为空")
                failed_tiles += 1
        
        self.logger.info(f"拼接完成: 成功 {successful_tiles} 个瓦片, 失败 {failed_tiles} 个瓦片")
        
        if successful_tiles == 0:
            self.logger.error("没有成功拼接任何瓦片，返回空白图像")
            # 创建一个带有错误信息的图像
            from PIL import ImageDraw, ImageFont
            error_image = Image.new('RGB', (width, height), color='lightgray')
            draw = ImageDraw.Draw(error_image)
            try:
                font = ImageFont.load_default()
                draw.text((10, 10), "No tiles downloaded", fill='red', font=font)
            except:
                pass
            return error_image
        
        return stitched
    
    def crop_to_bbox(self, image: Image.Image, tiles: List[Tuple[int, int]], zoom: int, 
                     north: float, south: float, east: float, west: float) -> Image.Image:
        """
        将拼接图像裁剪到精确的边界框
        
        Args:
            image: 拼接后的图像
            tiles: 瓦片坐标列表
            zoom: 缩放级别
            north, south, east, west: 目标边界框
            
        Returns:
            裁剪后的图像
        """
        if not tiles:
            return image
        
        # 计算瓦片范围
        x_coords = [x for x, y in tiles]
        y_coords = [y for x, y in tiles]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # 计算目标边界框在图像中的像素坐标
        target_x_min, target_y_max = self.deg2num(north, west, zoom)
        target_x_max, target_y_min = self.deg2num(south, east, zoom)
        
        # 转换为相对于拼接图像的像素坐标
        pixel_x_min = (target_x_min - min_x) * self.tile_size
        pixel_y_min = (target_y_min - min_y) * self.tile_size
        pixel_x_max = (target_x_max - min_x + 1) * self.tile_size
        pixel_y_max = (target_y_max - min_y + 1) * self.tile_size
        
        # 确保坐标在图像范围内
        pixel_x_min = max(0, pixel_x_min)
        pixel_y_min = max(0, pixel_y_min)
        pixel_x_max = min(image.width, pixel_x_max)
        pixel_y_max = min(image.height, pixel_y_max)
        
        # 裁剪图像
        if pixel_x_max > pixel_x_min and pixel_y_max > pixel_y_min:
            return image.crop((pixel_x_min, pixel_y_min, pixel_x_max, pixel_y_max))
        else:
            return image
    
    async def download_satellite_image(self, north: float, south: float, east: float, west: float, 
                                     zoom: int = None, output_path: str = None, 
                                     include_annotation: bool = False) -> Image.Image:
        """
        下载指定区域的卫星图像
        
        Args:
            north, south, east, west: 边界框坐标
            zoom: 缩放级别，如果不指定则使用最大级别
            output_path: 输出文件路径，如果不指定则不保存
            include_annotation: 是否包含注记图层
            
        Returns:
            下载的卫星图像
        """
        if zoom is None:
            zoom = self.max_zoom
        
        zoom = max(self.min_zoom, min(self.max_zoom, zoom))
        
        self.logger.info(f"开始下载卫星图像: 边界框({north}, {south}, {east}, {west}), 缩放级别: {zoom}")
        
        # 计算需要的瓦片
        tiles = self.calculate_tiles_for_bbox(north, south, east, west, zoom)
        self.logger.info(f"需要下载 {len(tiles)} 个瓦片")
        
        if len(tiles) > 100:
            self.logger.warning(f"瓦片数量较多({len(tiles)})，下载可能需要较长时间")
        
        # 下载卫星图像瓦片
        start_time = time.time()
        satellite_tiles = await self.download_tiles_batch(tiles, zoom, "satellite")
        
        # 如果需要注记图层，也下载注记瓦片
        if include_annotation:
            annotation_tiles = await self.download_tiles_batch(tiles, zoom, "annotation")
        
        download_time = time.time() - start_time
        self.logger.info(f"瓦片下载完成，耗时: {download_time:.2f}秒")
        
        # 拼接卫星图像
        satellite_image = self.stitch_tiles(satellite_tiles, tiles)
        
        # 如果有注记图层，进行合成
        if include_annotation:
            annotation_image = self.stitch_tiles(annotation_tiles, tiles)
            # 将注记图层叠加到卫星图像上
            satellite_image = Image.alpha_composite(
                satellite_image.convert('RGBA'),
                annotation_image.convert('RGBA')
            ).convert('RGB')
        
        # 裁剪到精确边界框
        final_image = self.crop_to_bbox(satellite_image, tiles, zoom, north, south, east, west)
        
        self.logger.info(f"最终图像尺寸: {final_image.size}")
        
        # 保存图像
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            final_image.save(output_path, quality=95)
            self.logger.info(f"卫星图像已保存到: {output_path}")
        
        return final_image

# 导入必要的模块
import io

# 便捷函数
async def download_tianditu_satellite(coordinates: List[Tuple[float, float]], 
                                    output_dir: str, 
                                    task_id: str,
                                    zoom: int = None,
                                    include_annotation: bool = False) -> str:
    """
    便捷函数：根据坐标下载天地图卫星图像
    
    Args:
        coordinates: 四个角点的经纬度坐标 [(lat1, lon1), (lat2, lon2), (lat3, lon3), (lat4, lon4)]
        output_dir: 输出目录
        task_id: 任务ID
        zoom: 缩放级别
        include_annotation: 是否包含注记
        
    Returns:
        保存的图像文件路径
    """
    # 设置日志
    logger = logging.getLogger(__name__)
    logger.info(f"开始天地图卫星图像下载任务: {task_id}")
    logger.info(f"输入坐标: {coordinates}")
    
    # 计算边界框
    lats = [coord[0] for coord in coordinates]
    lons = [coord[1] for coord in coordinates]
    
    north = max(lats)
    south = min(lats)
    east = max(lons)
    west = min(lons)
    
    logger.info(f"计算得到边界框: 北={north}, 南={south}, 东={east}, 西={west}")
    
    # 验证边界框
    if north <= south or east <= west:
        raise ValueError(f"无效的边界框: 北={north}, 南={south}, 东={east}, 西={west}")
    
    # 创建下载器
    downloader = TianDiTuDownloader()
    logger.info(f"使用API密钥: {downloader.api_key[:10]}..." if downloader.api_key else "未设置API密钥")
    
    # 设置输出路径
    output_path = os.path.join(output_dir, task_id, "satellite_image.jpg")
    logger.info(f"输出路径: {output_path}")
    
    # 设置合理的缩放级别
    if zoom is None:
        zoom = 18  # 降低默认缩放级别，避免瓦片过多
    zoom = max(18, min(18, zoom))  # 限制在合理范围内
    
    logger.info(f"使用缩放级别: {zoom}")
    
    # 下载图像
    try:
        image = await downloader.download_satellite_image(
            north=north, 
            south=south, 
            east=east, 
            west=west,
            zoom=zoom,
            output_path=output_path,
            include_annotation=include_annotation
        )
        
        # 验证下载的图像
        if image and image.size[0] > 0 and image.size[1] > 0:
            logger.info(f"天地图卫星图像下载成功: {output_path}, 尺寸: {image.size}")
            return output_path
        else:
            logger.error("下载的图像无效或为空")
            raise ValueError("下载的图像无效或为空")
            
    except Exception as e:
        logger.error(f"天地图卫星图像下载失败: {e}")
        raise
