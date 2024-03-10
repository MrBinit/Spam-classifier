# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install Flask and other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the label_encoder.pkl file into the container at /app
COPY Label_encoder.pkl /app

# Copy the tfid_vectorizer_encoder.pkl file into the container at /app
COPY tfidf_vectorizer.pkl /app
# Expose the Flask port
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
