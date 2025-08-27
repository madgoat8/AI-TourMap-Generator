"""
HuggingFace 镜像配置脚本
提供多个国内镜像源选择，解决网络访问问题
"""

import os
import sys

def set_mirror(mirror_name="hf-mirror"):
    """
    设置 HuggingFace 镜像源
    
    可选镜像:
    - hf-mirror: https://hf-mirror.com (推荐)
    - modelscope: https://www.modelscope.cn (阿里云)
    - official: https://huggingface.co (官方源)
    """
    
    mirrors = {
        "hf-mirror": "https://hf-mirror.com",
        "modelscope": "https://www.modelscope.cn", 
        "official": "https://huggingface.co"
    }
    
    if mirror_name not in mirrors:
        print(f"❌ 未知镜像: {mirror_name}")
        print(f"可选镜像: {', '.join(mirrors.keys())}")
        return False
    
    mirror_url = mirrors[mirror_name]
    os.environ['HF_ENDPOINT'] = mirror_url
    
    print(f"✅ 已设置镜像源: {mirror_name}")
    print(f"📡 镜像地址: {mirror_url}")
    return True

def test_mirror_connectivity():
    """测试镜像连通性"""
    import requests
    from urllib.parse import urljoin
    
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
    test_url = urljoin(endpoint, '/api/models')
    
    try:
        print(f"🔍 测试镜像连通性: {endpoint}")
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            print("✅ 镜像连接正常")
            return True
        else:
            print(f"⚠️ 镜像响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 镜像连接失败: {e}")
        return False

def auto_select_mirror():
    """自动选择最佳镜像"""
    mirrors = ["hf-mirror", "modelscope", "official"]
    
    print("🔄 自动选择最佳镜像...")
    
    for mirror in mirrors:
        print(f"\n📡 尝试镜像: {mirror}")
        if set_mirror(mirror):
            if test_mirror_connectivity():
                print(f"🎯 选择镜像: {mirror}")
                return mirror
    
    print("❌ 所有镜像都无法连接")
    return None

def show_mirror_info():
    """显示镜像信息"""
    print("🌐 可用镜像源:")
    print("1. hf-mirror   - https://hf-mirror.com (推荐，国内优化)")
    print("2. modelscope  - https://www.modelscope.cn (阿里云)")
    print("3. official    - https://huggingface.co (官方源)")
    print()
    print("💡 使用方法:")
    print("python setup_mirror.py hf-mirror")
    print("python setup_mirror.py auto  # 自动选择")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_mirror_info()
        sys.exit(0)
    
    mirror_choice = sys.argv[1].lower()
    
    if mirror_choice == "auto":
        selected_mirror = auto_select_mirror()
        if selected_mirror:
            print(f"\n🎉 自动选择完成，当前镜像: {selected_mirror}")
        else:
            print("\n💔 自动选择失败，请手动指定镜像")
            sys.exit(1)
    elif mirror_choice == "test":
        test_mirror_connectivity()
    elif mirror_choice == "info":
        show_mirror_info()
    else:
        if set_mirror(mirror_choice):
            test_mirror_connectivity()
        else:
            show_mirror_info()
            sys.exit(1)