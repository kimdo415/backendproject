FROM python:3.9-slim-buster
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY main.py /
COPY face.py /
COPY images/ /images/
COPY dyk_club/ /dyk_club/
COPY jordan.onnx /
COPY trained_knn_model.clf /
# 변경된 내용을 포함한 draw.py 파일을 추가
#COPY /home/zero/miniforge3/envs/zero/lib/python3.9/site-packages/cvu /usr/local/lib/python3.9/site-packages/cvu
COPY cvu /usr/local/lib/python3.9/site-packages/cvu
COPY .env /

EXPOSE 8989

# ENTRYPOINT ["python3", "main.py"]

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8989"]
