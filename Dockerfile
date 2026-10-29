FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV HTTP_PROXY=$HTTP_PROXY HTTPS_PROXY=$HTTPS_PROXY
RUN apt-get -o Acquire::http::Proxy=$HTTP_PROXY update && apt-get -o Acquire::http::Proxy=$HTTP_PROXY install -y --no-install-recommends libcairo2 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir numpy pillow opencv-python-headless scikit-image scipy "vtracer==1.0.0a3" cairosvg==2.7.1 flask gunicorn --pre
COPY lumina_vec /app/lumina_vec
COPY webui.py index.html /app/
RUN mkdir -p /app/weights /data/uploads /data/outputs
ARG MODEL_URL=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
RUN python -c "import urllib.request; urllib.request.urlretrieve('$MODEL_URL','/app/weights/realesr-animevideov3.pth')"
EXPOSE 8080
CMD ["gunicorn","-w","1","-b","0.0.0.0:8080","--timeout","600","webui:app"]
