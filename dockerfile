# Use the official Python image as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the FastAPI script and requirements file into the container
COPY ./app.py /app
COPY ./model_xgb.pkl /app

# Install dependencies
RUN pip install unicorn

# Expose the port that the FastAPI application will run on
EXPOSE 5000

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
