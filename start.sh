#!/bin/bash
# 容器启动脚本:同时跑 FastAPI(内部8000) 和 Streamlit(对外7860)

# 后台启动后端
python src/api.py > /tmp/api.log 2>&1 &

# 等后端起来(最多等30秒)
for i in $(seq 1 30); do
  if curl -s http://127.0.0.1:8000/docs > /dev/null 2>&1; then
    echo "后端已就绪"
    break
  fi
  sleep 1
done

# 前台启动前端(容器主进程,挂掉则容器退出)
# enableCORS/enableXsrfProtection 必须关闭:
#   平台通过 ms.show 域名反代到 FC 内部主机名,Origin 与 Host 不一致
#   Streamlit 默认的跨域保护会拒绝 WebSocket 连接 → 页面一直转圈
exec streamlit run src/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
