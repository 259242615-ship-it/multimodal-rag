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
exec streamlit run src/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true
