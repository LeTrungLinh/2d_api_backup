FROM python:3.10.9-slim-buster
ENV http_proxy "http://proxy.hcm.fpt.vn:80"
ENV https_proxy "http://proxy.hcm.fpt.vn:80"

WORKDIR /usr/src/app
    ENV TZ="Asia/Ho_Chi_Minh"
RUN apt-get update && \
    apt-get install -y python3-tk && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    python3 -m pip install --upgrade pip &&\
    apt-get install -y nano
COPY ./requirements.txt /usr/src/app
RUN python3 -m pip install --no-cache-dir --default-timeout=100 -r requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"
ENV PYTHONUNBUFFERED 1





COPY . /usr/src/app
ENV http_proxy ""
ENV https_proxy ""
# RUN mkdir -p /opt/pdx/mypt && chown 666 /opt/pdx/mypt   # Remove when deploy at ISC
EXPOSE 8000
CMD ["/bin/bash", "run.sh"]

