FROM continuumio/miniconda3
 
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
 
ENV CC=gcc
ENV CXX=g++
 
COPY environment.yml /tmp/environment.yml
RUN conda env update -n base -f /tmp/environment.yml
RUN pip install stilts-nli