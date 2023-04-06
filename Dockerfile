# create setup container
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu18.04 AS setup

# update
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends
RUN apt-get install -y git
COPY --from=ghcr.io/tarampampam/curl:8.0.1 /bin/curl /bin/curl

# setup python3.8
RUN apt-get install -y python3.8 python3.8-distutils python3.8-dev python3.8-venv python3-pip

# create venv
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# clone Real-ESRGAN
RUN git clone https://github.com/xinntao/Real-ESRGAN.git /usr/src/app
WORKDIR /usr/src/app

# setup python packages
RUN python3.8 -m pip install --upgrade pip==23.0.1
RUN python3.8 -m pip --no-cache-dir install torch==1.13.1 --extra-index-url=https://download.pytorch.org/whl/cu116
RUN python3.8 -m pip install runpod==0.9.1
RUN python3.8 -m pip install Cython
RUN python3.8 -m pip install basicsr
RUN python3.8 -m pip install facexlib
RUN python3.8 -m pip install gfpgan
RUN python3.8 -m pip install -r requirements.txt
RUN python3.8 setup.py develop

# download models
RUN curl -L -O --create-dirs --output-dir weights https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
RUN curl -L -O --create-dirs --output-dir weights https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth
RUN curl -L -O --create-dirs --output-dir weights https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
RUN curl -L -O --create-dirs --output-dir weights https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

# create runtime container
FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu18.04

# update
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends
RUN apt-get install -y python3.8 libglib2.0-0 libgl1-mesa-dev

# setup venv
ENV PATH=/venv/bin:$PATH
COPY --from=setup /venv /venv

# copy Real-ESRGAN
COPY --from=setup /usr/src/app /usr/src/app
WORKDIR /usr/src/app

ADD handler.py .
ADD test_input.json .

CMD [ "python3.8", "-u", "handler.py" ]