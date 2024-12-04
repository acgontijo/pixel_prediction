FROM python:3.10.6-buster

# Install dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && apt-get clean

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_VERSION=3.10.0

# Install Python dependencies
RUN pip install --upgrade pip
COPY requirements_app.txt requirements.txt
RUN pip install -r requirements.txt

# Copy application code
COPY setup.py setup.py
COPY /api /api

# Install the package
# RUN pip install .

# Run the API
CMD uvicorn api.api:app --host 0.0.0.0 --port ${PORT:-8000}
