# Use the official Python 3.11 image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system packages including CA certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip && pip install pip-tools

# Install certifi package to manage SSL certificates
RUN pip install certifi

# Building requirements.txt file from requirements.in file
COPY requirements.in .
RUN pip-compile --output-file requirements.txt requirements.in

# Install the necessary Python packages, including pystemmer with trusted hosts
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container except
COPY . .

# Initialize the SQLite database
RUN python src/pipelines/initialize_db.py

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
