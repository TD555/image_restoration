FROM ubuntu:22.04

RUN apt-get update -y && \
    apt-get install --fix-missing -y python3.9 python3-pip --no-install-recommends ffmpeg libsm6 libxext6 && \
    apt-get clean 

RUN adduser --disabled-login service-davtashen
USER service-davtashen

WORKDIR /var/www/davtashen/davtashen-service 

COPY --chown=service-davtashen:service-davtashen .   /var/www/davtashen/davtashen-service 

COPY --chown=service-davtashen:service-davtashen ./requirements.txt  /var/www/davtashen/davtashen-service/requirements.txt

ENV PATH="$PATH:/home/service-davtashen/.local/bin"

RUN pip3 install torch==2.0.0+cpu torchvision==0.15.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN python3 -m pip install --no-cache-dir -r requirements.txt && rm -rf ~/.cache/pip

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "5"]