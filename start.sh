#!/bin/bash
# 容器启动脚本:同时跑 FastAPI(内部8000) 和 Streamlit(对外7860)

echo "=========== 环境自检 ==========="
echo "工作目录: $(pwd)"
echo "目录内容:"
ls -la /app
echo "向量库:"
ls -la /app/chroma_db_multimodal || echo "!! 向量库目录不存在"
if [ -n "$BAILIAN_API_KEY" ]; then
  echo "BAILIAN_API_KEY: 已注入(长度 ${#BAILIAN_API_KEY})"
else
  echo "!! BAILIAN_API_KEY 未注入"
fi
echo "==============================="

# 后台启动后端(日志不重定向,直接进容器日志便于排查)
python src/api.py &
API_PID=$!
echo "FastAPI 已启动 PID=$API_PID"

# 等后端就绪(镜像里没有 curl,用 python 检测)
for i in $(seq 1 30); do
  if python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/docs',timeout=2)" >/dev/null 2>&1; then
    echo "后端已就绪"
    break
  fi
  sleep 1
done

# 前台启动前端(容器主进程)
# enableCORS/enableXsrfProtection 必须关闭:
#   平台通过 ms.show 域名反代到 FC 内部主机名,Origin 与 Host 不一致
#   Streamlit 默认的跨域保护会拒绝 WebSocket → 页面一直转圈
exec streamlit run src/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
