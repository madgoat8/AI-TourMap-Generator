"""
OSM数据获取客户端
"""
import requests
import logging
from typing import Dict, Any
from config import OVERPASS_URL

logger = logging.getLogger(__name__)

class OSMClient:
    """OpenStreetMap数据获取客户端"""
    
    def __init__(self, base_url: str = OVERPASS_URL):
        self.base_url = base_url
    
    def build_query(self, bbox_str: str) -> str:
        """
        构建Overpass查询语句 - 简化版本，只查询道路和水面
        
        Args:
            bbox_str: 边界框字符串，格式为 "min_lat,min_lng,max_lat,max_lng"
            
        Returns:
            Overpass查询语句
        """
        query = f'''[out:json][timeout:25];
        (
          way["highway"]({bbox_str}); 
          way["natural"="water"]({bbox_str});
          way["waterway"]({bbox_str}); 
        );
        out body;>;out skel qt;'''
        return query
    
    def fetch_data(self, bbox_str: str) -> Dict[str, Any]:
        """
        获取指定区域的OSM数据
        
        Args:
            bbox_str: 边界框字符串
            
        Returns:
            OSM数据字典
            
        Raises:
            requests.RequestException: 网络请求失败
            ValueError: 数据解析失败
        """
        query = self.build_query(bbox_str)
        logger.info(f"正在查询OSM数据，边界框: {bbox_str}")
        
        try:
            response = requests.post(self.base_url, data=query, timeout=30)
            response.raise_for_status()
            osm_data = response.json()
            
            element_count = len(osm_data.get('elements', []))
            logger.info(f"成功获取 {element_count} 个地图元素")
            
            return osm_data
            
        except requests.RequestException as e:
            logger.error(f"OSM数据请求失败: {e}")
            raise
        except ValueError as e:
            logger.error(f"OSM数据解析失败: {e}")
            raise
    
    def get_statistics(self, osm_data: Dict[str, Any]) -> Dict[str, int]:
        """
        获取OSM数据统计信息 - 简化版本，只统计道路和水面
        
        Args:
            osm_data: OSM数据字典
            
        Returns:
            统计信息字典
        """
        elements = osm_data.get('elements', [])
        ways = [elem for elem in elements if elem['type'] == 'way']
        
        stats = {
            'total_elements': len(elements),
            'total_ways': len(ways),
            'highways': len([w for w in ways if 'highway' in w.get('tags', {})]),
            'water_bodies': len([w for w in ways if 'natural' in w.get('tags', {}) and w['tags']['natural'] == 'water']),
            'waterways': len([w for w in ways if 'waterway' in w.get('tags', {})])
        }
        
        return stats