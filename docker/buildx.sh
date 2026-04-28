#!/bin/bash
set -e

BUILDER_NAME="limited-builder"

# 1. 删除旧 builder（安全忽略错误）
docker buildx rm "$BUILDER_NAME" 2>/dev/null || true

# 2. 创建新 builder
docker buildx create \
  --name "$BUILDER_NAME" \
  --use \
  --driver docker-container \
  --driver-opt image=moby/buildkit:v0.24.0 \
  --buildkitd-flags '--allow-insecure-entitlement security.insecure --allow-insecure-entitlement network.host'

# 3. 启动并等待就绪（bootstrap 会确保容器运行并连接成功）
docker buildx inspect "$BUILDER_NAME" --bootstrap >/dev/null

# 4. 获取容器名并限制资源
CONTAINER_NAME=$(docker ps -f "name=buildx_buildkit_${BUILDER_NAME}" --format "{{.Names}}" | head -n1)
if [ -n "$CONTAINER_NAME" ]; then
  docker update --memory=16g --memory-swap=16g --cpus=10 "$CONTAINER_NAME" >/dev/null
  echo "✅ Builder '$BUILDER_NAME' is ready with 16G memory (no swap) and 10 CPUs."
else
  echo "❌ Failed to find BuildKit container." >&2
  exit 1
fi