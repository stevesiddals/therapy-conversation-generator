
# 1. Base Image: Start with a reliable, lightweight Python image
# We recommend a specific version like 3.11-slim for better stability and smaller size
FROM python:3.11-slim

# 2. Set Working Directory: This is the directory inside the container where your app will live
WORKDIR /app

# 3. Copy Requirements and Install Dependencies:
# We copy requirements.txt first to leverage Docker's build cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Application Code: Copy everything else in your project folder
# This includes your app.py, your .env file (which will be ignored since we use secrets), etc.
COPY . .

# 5. Configuration: Set the environment variable Cloud Run needs (PORT)
# Cloud Run expects the app to listen on the port specified by the $PORT environment variable,
# which defaults to 8080 if not set.
ENV PORT 8080
EXPOSE 8080

# 6. Run Command: The final command to start the Streamlit application
# We tell Streamlit to listen on all interfaces (0.0.0.0) and on port 8080 (which is $PORT)
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]