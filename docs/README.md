# AI-TourMap-Generator 文档中心

欢迎使用 AI-TourMap-Generator 项目文档！这里包含了项目的详细使用指南和技术文档。

## 📚 文档目录

### 🔽 模型下载相关

| 文档 | 描述 | 适用场景 |
|------|------|---------|
| [模型下载指南](./模型下载指南.md) | 📖 详细的模型下载指南（中文） | 完整配置和故障排除 |
| [Model Download Guide](./Model-Download-Guide.md) | 📖 Comprehensive model download guide (English) | Complete setup and troubleshooting |
| [快速参考](./快速参考.md) | ⚡ 快速参考卡片 | 日常使用和快速查询 |

### 🛠️ 相关工具

| 工具 | 功能 | 位置 |
|------|------|------|
| `download_models.py` | 模型下载脚本 | 项目根目录 |
| `setup_mirror.py` | 镜像配置工具 | 项目根目录 |
| `download_models.bat` | Windows一键下载 | 项目根目录 |

## 🚀 快速开始

### 新用户推荐流程

1. **第一次使用**
   ```bash
   # Windows
   download_models.bat
   
   # Linux/Mac
   python download_models.py
   ```

2. **遇到网络问题**
   ```bash
   python setup_mirror.py auto
   ```

3. **离线环境部署**
   ```bash
   start_offline.bat
   ```

### 常用命令速查

```bash
# 镜像配置
python setup_mirror.py hf-mirror  # 设置国内镜像
python setup_mirror.py test       # 测试连通性
python setup_mirror.py auto       # 自动选择

# 模型下载
python download_models.py         # 标准下载
download_models.bat               # Windows一键下载

# 启动服务
start.bat                         # 标准启动
start_gpu.bat                     # GPU加速启动
start_offline.bat                 # 离线启动
```

## 📊 性能参考

### 镜像速度对比

| 镜像源 | 国内速度 | 海外速度 | 稳定性 |
|--------|---------|---------|-------|
| hf-mirror | 🚀🚀🚀🚀🚀 | 🚀🚀🚀 | ⭐⭐⭐⭐⭐ |
| modelscope | 🚀🚀🚀🚀 | 🚀🚀 | ⭐⭐⭐⭐ |
| official | 🚀 | 🚀🚀🚀🚀🚀 | ⭐⭐⭐ |

### 系统要求

- **存储空间**: 最少 10GB，推荐 20GB
- **网络带宽**: 建议 10Mbps+
- **内存**: 下载时需要 2GB+

## 🔧 故障排除快速索引

| 错误类型 | 快速解决 | 详细文档 |
|---------|---------|---------|
| 网络连接失败 | `python setup_mirror.py auto` | [镜像配置](./模型下载指南.md#镜像源配置) |
| 代理配置冲突 | 清除代理变量 | [故障排除](./模型下载指南.md#故障排除) |
| 磁盘空间不足 | 清理缓存目录 | [高级配置](./模型下载指南.md#高级配置) |
| 权限问题 | `chmod 755 model_cache` | [故障排除](./模型下载指南.md#故障排除) |

## 📞 获取帮助

- 📖 **详细文档**: 查看对应的详细指南
- 🐛 **问题报告**: 创建 GitHub Issue  
- 💬 **讨论交流**: 项目 Wiki 或交流群
- ⚡ **快速查询**: 使用快速参考卡片

## 📋 文档更新日志

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0 | 2024-08 | 初始版本，包含完整的模型下载指南 |

---

**维护者**: AI-TourMap-Generator Team  
**最后更新**: 2024年8月