# Use official lightweight Python image
FROM python:3.11-slim

# Prevents Python from writing pyc files to disk and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory inside container
WORKDIR /app

# Copy only requirements.txt first (for caching)
COPY requirements.txt /app/

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy rest of the application code
COPY . /app/

# Expose port 5000 for the Flask app
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
