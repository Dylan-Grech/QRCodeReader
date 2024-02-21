# Use the official Python image from the Docker Hub
FROM python:3.11.4

# Install required system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y libgl1-mesa-glx libzbar-dev

# Set the working directory inside the container
WORKDIR /app

# Copy the local code to the container
COPY . .

# Install any dependencies your Python application requires
RUN pip install -r requirements.txt  # If you have a requirements.txt file

# Specify the command to run your application
CMD ["python", "main.py"]
