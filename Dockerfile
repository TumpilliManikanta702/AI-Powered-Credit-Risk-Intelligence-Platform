FROM python:3.10-slim

WORKDIR /app

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

# Generate dummy data and train model during build so container runs out of the box
RUN python src/utils/helpers.py
RUN python src/ml/train.py
RUN python src/data/loader.py

EXPOSE 8501

CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
