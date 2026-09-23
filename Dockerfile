FROM python:3.12-slim

WORKDIR /app

# 先只拷依赖清单 → 装依赖(利用Docker层缓存,改代码时不用重装)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/

# 再拷项目代码
COPY . .

# 平台要求的对外端口
EXPOSE 7860

CMD ["bash", "start.sh"]
