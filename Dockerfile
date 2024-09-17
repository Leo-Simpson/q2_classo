# syntax=docker/dockerfile:1
FROM quay.io/qiime2/core:2021.4

COPY . ./
WORKDIR .
RUN python -m pip install -r requirements.txt

