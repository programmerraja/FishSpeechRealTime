FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    scipy \
    uv \
    numpy \
    websockets \
    asyncio

ENV ORPHEUS_CPP_VERBOSE=true
ENV ORPHEUS_CPP_LANG=en
ENV HF_HOME=/root/.cache/huggingface/hub

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .


EXPOSE 9802

CMD ["python", "server.py"]