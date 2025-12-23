# ER-NeRF Docker 封装文档索引

本目录包含ER-NeRF Docker封装的完整文档。

## 📚 文档列表

### 快速开始

1. **[完整部署指南](ERNERF_DEPLOYMENT.md)** ⭐
   - 系统要求
   - 快速部署
   - 详细步骤
   - 故障排查
   - 性能优化

2. **[Docker使用指南](ERNERF_DOCKER.md)**
   - Docker命令详解
   - 工作流程说明
   - 预处理/训练/推理步骤
   - 系统架构

3. **[Backend集成指南](ERNERF_BACKEND_INTEGRATION.md)**
   - Docker客户端使用
   - 环境变量配置
   - API调用示例
   - 路径映射说明

## 🚀 快速导航

### 按使用场景

| 场景 | 推荐文档 |
|------|----------|
| **首次部署** | [完整部署指南](ERNERF_DEPLOYMENT.md) |
| **日常使用** | [Docker使用指南](ERNERF_DOCKER.md) |
| **开发集成** | [Backend集成指南](ERNERF_BACKEND_INTEGRATION.md) |
| **遇到问题** | [完整部署指南 > 故障排查](ERNERF_DEPLOYMENT.md#故障排查) |

### 按角色

| 角色 | 推荐阅读 |
|------|----------|
| **系统管理员** | 部署指南 + 故障排查 |
| **开发者** | Backend集成 + API文档 |
| **研究人员** | Docker使用指南 + 论文 |
| **普通用户** | 快速开始 + 常见问题 |

## 📋 检查清单

### 部署前检查

- [ ] 确认系统为Linux（Ubuntu 18.04+）
- [ ] 确认有NVIDIA GPU（≥24GB显存）
- [ ] 安装Docker和NVIDIA Container Toolkit
- [ ] 准备BFM模型
- [ ] 有足够的磁盘空间（≥100GB）

### 部署步骤

1. [ ] 克隆项目代码
2. [ ] 准备BFM模型
3. [ ] 构建Docker镜像（30-60分钟）
4. [ ] 启动容器
5. [ ] 验证安装
6. [ ] 测试预处理

### 集成步骤

1. [ ] 设置环境变量 `USE_DOCKER_FOR_ERNERF=true`
2. [ ] 测试Backend API
3. [ ] 上传测试视频
4. [ ] 测试训练流程
5. [ ] 测试推理流程

## 🔧 核心概念

### Docker架构

```
Mac/开发机器              Linux服务器(有GPU)
┌─────────────┐           ┌──────────────────┐
│  Flask Web  │  ←──API──→│  ER-NeRF Docker  │
│   Service   │           │     Container     │
└─────────────┘           └──────────────────┘
```

### 数据流

```
用户上传视频
    ↓
Backend接收请求
    ↓
调用Docker客户端
    ↓
Docker容器预处理
    ↓
Docker容器训练
    ↓
用户上传音频
    ↓
Docker容器推理
    ↓
返回生成视频
```

## 📦 文件说明

### Docker相关文件

| 文件 | 说明 |
|------|------|
| `Dockerfile.ernerf` | ER-NeRF Docker镜像定义 |
| `docker-compose.ernerf.yml` | Docker Compose配置 |
| `.dockerignore` | Docker构建忽略文件 |
| `docker/ernerf/scripts/entrypoint.sh` | 容器启动脚本 |

### Backend相关文件

| 文件 | 说明 |
|------|------|
| `backend/ernerf_docker_client.py` | ER-NeRF Docker客户端 |
| `backend/video_generator.py` | 视频生成（已集成Docker） |
| `backend/model_trainer.py` | 模型训练（已集成Docker） |

### 脚本文件

| 文件 | 说明 |
|------|------|
| `scripts/start_ernerf_docker.sh` | 快速启动脚本 |

## ⚡ 常用命令

```bash
# 构建镜像
docker compose -f docker-compose.ernerf.yml build

# 启动服务
docker compose -f docker-compose.ernerf.yml up -d

# 查看日志
docker compose -f docker-compose.ernerf.yml logs -f

# 停止服务
docker compose -f docker-compose.ernerf.yml down

# 进入容器
docker compose -f docker-compose.ernerf.yml run --rm ernerf shell

# 预处理
docker compose -f docker-compose.ernerf.yml run --rm ernerf preprocess /path/to/video.mp4 task_id

# 训练
docker compose -f docker-compose.ernerf.yml run --rm ernerf train data/task models/ER-NeRF/task

# 推理
docker compose -f docker-compose.ernerf.yml run --rm ernerf test data/task models/ER-NeRF/task audio.npy
```

## 🐛 故障排查

### 快速诊断

```bash
# 1. 检查Docker
docker --version
docker ps

# 2. 检查NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu18.04 nvidia-smi

# 3. 检查容器状态
docker compose -f docker-compose.ernerf.yml ps

# 4. 查看日志
docker compose -f docker-compose.ernerf.yml logs --tail=100 ernerf

# 5. 进入容器调试
docker compose -f docker-compose.ernerf.yml run --rm ernerf shell
```

### 常见问题

1. **构建失败** → 检查网络、清理缓存
2. **CUDA错误** → 重新安装nvidia-container-toolkit
3. **内存不足** → 减小batch size或使用更大GPU
4. **文件找不到** → 检查卷挂载路径

详细解决方案见[完整部署指南 > 故障排查](ERNERF_DEPLOYMENT.md#故障排查)

## 📞 获取帮助

### 文档查找

1. 使用 `Ctrl+F` 搜索关键词
2. 查看对应文档的目录
3. 阅读故障排查章节

### 提交问题

提交Issue时请包含：
- 系统信息（`uname -a`）
- Docker版本（`docker --version`）
- GPU信息（`nvidia-smi`）
- 错误日志
- 复现步骤

## 🔄 更新日志

### v1.0 (2024-12-24)

- ✅ 完整的Docker封装
- ✅ 支持训练和推理
- ✅ Backend集成
- ✅ 多GPU支持
- ✅ 自动化脚本
- ✅ 完整文档

## 📈 后续计划

- [ ] 添加分布式训练支持
- [ ] 优化构建速度
- [ ] 添加更多预训练模型
- [ ] 性能监控面板
- [ ] 自动扩缩容

---

**注意**: 本Docker封装仅支持Linux系统，不支持macOS和Windows。如需要在Mac上开发，请参考[Backend集成指南](ERNERF_BACKEND_INTEGRATION.md)中的远程部署方案。
