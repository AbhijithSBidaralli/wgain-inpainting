FROM pytorch/pytorch

COPY MMEmotionRecognition/requirements.txt .

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get install -y libsndfile1
RUN pip install soundfile

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip uninstall -y torch torchvision torchaudio
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

CMD ["echo","Image created!"] 