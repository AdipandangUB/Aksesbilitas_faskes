#!/bin/bash

# Install system dependencies for pyproj
apt-get update && apt-get install -y \
    libproj-dev \
    proj-bin \
    proj-data \
    libgeos-dev \
    libspatialindex-dev

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt