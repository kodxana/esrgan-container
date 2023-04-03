FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu18.04

# update
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends
RUN apt-get install -y git software-properties-common
COPY --from=ghcr.io/tarampampam/curl:8.0.1 /bin/curl /bin/curl

# setup python3.10
RUN apt-get install -y python3.8 python3.8-distutils python3.8-dev python3.8-venv python3-pip

# create venv
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# clone Real-ESRGAN
RUN git clone https://github.com/xinntao/Real-ESRGAN.git /usr/src/app

WORKDIR /usr/src/app

# upgrade pip
RUN python3.8 -m pip install --upgrade pip==23.0.1
# install runpod
RUN python3.8 -m pip install runpod==0.9.1
RUN python3.8 -m pip install cog
# install Real-ESRGAN dependencies
RUN apt-get install -y libgl1-mesa-dev
RUN python3.8 -m pip install Cython
RUN python3.8 -m pip install torch==1.13.1 --extra-index-url=https://download.pytorch.org/whl/cu116
RUN python3.8 -m pip install basicsr
RUN python3.8 -m pip install facexlib
RUN python3.8 -m pip install gfpgan
RUN python3.8 -m pip install -r requirements.txt
RUN python3.8 setup.py develop

ADD handler.py .
ADD test_input.json .

RUN curl -L -O --create-dirs --output-dir weights https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

CMD [ "python3.8", "-u", "handler.py" ]