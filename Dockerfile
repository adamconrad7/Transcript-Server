# Use NVIDIA PyTorch container as base
#FROM nvcr.io/nvidia/pytorch:24.02-py3
#FROM nvcr.io/nvidia/nemo:dev
FROM nvcr.io/nvidia/nemo:24.05
# Install NeMo toolkit
#RUN pip install nemo_toolkit[all]

# Install any additional requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set up your working directory
WORKDIR /app

# Copy your source code
COPY . .

# Command to run your application

CMD ["python", "-m", "src.main"]
#CMD ["/bin/sh", "-c", "bash"]
