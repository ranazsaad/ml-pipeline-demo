# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Accept RUN_ID as build argument
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

# Install required packages
RUN pip install mlflow scikit-learn pandas numpy

# Create a script to download the model
RUN echo '#!/bin/bash\n\
echo "==================================="\n\
echo "📦 Downloading model for Run ID: $RUN_ID"\n\
echo "==================================="\n\
echo "This is a mock build. In production, you would:"\n\
echo "1. Connect to MLflow server"\n\
echo "2. Download model artifacts"\n\
echo "3. Prepare model for serving"\n\
echo "4. Start model server"\n\
echo "==================================="\n\
echo "✅ Model downloaded successfully!"\n\
' > /app/download_model.sh && chmod +x /app/download_model.sh

# Command to run when container starts
CMD ["/bin/bash", "-c", "/app/download_model.sh && echo \"🚀 Container ready for model serving\""]