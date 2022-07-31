FROM tensorflow/tensorflow

COPY inpainting/requirments.txt .

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirments.txt

CMD ["echo","Image created!"] 