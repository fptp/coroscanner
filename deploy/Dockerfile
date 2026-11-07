# Image du worker CoroSeg <-> Orthanc.
#   GPU (défaut) : docker build -f deploy/Dockerfile -t coroseg-orthanc .
#   CPU          : docker build -f deploy/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t coroseg-orthanc .
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOTALSEG_HOME_DIR=/data/totalseg \
    COROSEG_WORK_DIR=/data/coroseg

RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-dejavu-core libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install torch --index-url ${TORCH_INDEX}
COPY requirements-orthanc.txt /app/
RUN pip install -r /app/requirements-orthanc.txt

COPY coroseg /app/coroseg
COPY coroseg_orthanc /app/coroseg_orthanc
COPY deploy/entrypoint.sh /app/entrypoint.sh
WORKDIR /app
VOLUME ["/data"]
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["watch"]
