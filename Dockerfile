FROM python:3.10-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Install system dependencies for LightGBM and others
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Pipeline will run at startup if models don't exist yet (important for volume mounts)
EXPOSE 8501

CMD bash -c "python src/utils/helpers.py && \
    if [ ! -f 'models/model.pkl' ]; then \
        python src/ml/train.py && \
        python src/data/loader.py; \
    fi && \
    streamlit run src/ui/app.py --server.port=8501 --server.address=0.0.0.0"
