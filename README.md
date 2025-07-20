# Elementary-LSTM-Based-Sentiment-Analysis-Pipeline

🚀 Sentiment Analysis Pipeline with PyTorch, FastAPI & Docker
This project demonstrates a complete end-to-end pipeline for sentiment analysis, from data ingestion and deep learning model training to API deployment using Docker. It leverages PyTorch for the deep learning model, FastAPI for building a robust API, and PostgreSQL for data storage.

✨ Project Goal
The primary goal of this project is to provide a practical, hands-on example of building a machine learning pipeline. It covers:

Data Acquisition & Storage: Downloading a public dataset and storing it in a relational database.

Model Training: Implementing and training a deep learning model (using PyTorch and Hugging Face Transformers) for sentiment classification.

API Development: Creating a RESTful API with FastAPI to serve model predictions.

Containerization & Deployment: Packaging the application using Docker for consistent and reproducible deployment.

🌟 Features
Automated Data Ingestion: Python script to download the IMDB movie review dataset and populate a PostgreSQL database.

PyTorch Deep Learning Model: Utilizes the Hugging Face transformers library for fine-tuning a pre-trained model for sentiment analysis.

High-Performance API: Built with FastAPI for fast and asynchronous request handling.

Dockerized Deployment: Ensures the application runs consistently across different environments.

Clear Project Structure: Organized into logical modules for better maintainability.

🛠️ Technologies Used
Python 3.11.13: Core programming language.

PostgreSQL: Relational database for storing review data.

PyTorch: Deep learning framework.

Hugging Face Transformers: For pre-trained models and tokenizers.

FastAPI: Web framework for building the prediction API.

Uvicorn: ASGI server for running FastAPI.

Docker: For containerization.

psycopg2-binary: PostgreSQL adapter for Python.

datasets: For easily loading public datasets.

pandas: Data manipulation and analysis.

scikit-learn: For data splitting and evaluation utilities.

📁 Folder Structure
sentiment_analysis/
├── connections/
│   └── postgres_connection.py  # (Placeholder for future database connection logic)
├── data/
│   └── postgres_queries.py     # (Placeholder for future database query logic)
├── scripts/
│   └── data_ingestion.py       # Downloads and stores data in PostgreSQL
│   └── model_training.py       # Fetches data, trains PyTorch model, saves it
├── src/
│   ├── app.py                  # FastAPI application for predictions
│   └── trained_pytorch_model/  # Directory where trained PyTorch model & tokenizer are saved
│       ├── pytorch_model.bin
│       ├── config.json
│       └── tokenizer.json (or similar tokenizer files)
├── Dockerfile                  # Docker build instructions
└── requirements.txt            # Python dependencies
└── README.md                   # This project introduction

🚀 Getting Started
Follow these steps to set up and run the project locally.

Prerequisites

Docker: Install Docker Desktop

Python 3.11.13: Install Python

PostgreSQL: Install PostgreSQL

1. Database Setup

Install PostgreSQL: If you haven't already, install PostgreSQL on your system.

Create Database and User:
Open your terminal and use psql to create a new user and database:

sudo -i -u postgres
psql
CREATE USER your_db_user WITH PASSWORD 'your_strong_password';
CREATE DATABASE sentiment_db OWNER your_db_user;
\q
exit

Replace your_db_user and your_strong_password with your desired credentials. Remember these!

Set Environment Variables:
Before running Python scripts, set these environment variables (or hardcode them directly in the scripts for quick testing, but environment variables are recommended for production):

export DB_NAME="sentiment_db"
export DB_USER="your_db_user"
export DB_PASSWORD="your_strong_password"
export DB_HOST="localhost"
export DB_PORT="5432"

2. Python Environment Setup

Clone the repository:

git clone https://github.com/your-username/sentiment_analysis.git
cd sentiment_analysis

Create a virtual environment:

python3.11 -m venv venv

Activate the virtual environment:

On macOS/Linux:

source venv/bin/activate

On Windows:

.\venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

3. Data Ingestion

Run the script to download the IMDB dataset and store it in your PostgreSQL database:

python scripts/data_ingestion.py

4. Model Training

Run the script to train the PyTorch sentiment analysis model. This script will save the trained model and tokenizer into the src/trained_pytorch_model/ directory.

Note: You will need to adapt scripts/model_training.py to use PyTorch and save the model/tokenizer using save_pretrained() method from Hugging Face Transformers. A basic example of how to save is provided in the src/app.py comments.

python scripts/model_training.py

Verify that src/trained_pytorch_model/ now contains the necessary model and tokenizer files (e.g., pytorch_model.bin, config.json, tokenizer.json).

5. Dockerize and Run the API

Build the Docker image:
Navigate to the root sentiment_analysis/ directory (where Dockerfile is located) and build the image:

docker build -t sentiment-predictor-pytorch .

Run the Docker container:
This will start the FastAPI application, mapping port 5000 from the container to port 5000 on your host machine:

docker run -p 5000:5000 sentiment-predictor-pytorch

The API should now be accessible at http://localhost:5000.

💡 Usage
Once the Docker container is running, you can interact with the API:

Health Check

Endpoint: GET /health

Description: Checks if the API is running and if the model is loaded.

Example curl command:

curl http://localhost:5000/health

Expected Response:

{
  "status": "healthy",
  "model_loaded": true,
  "tokenizer_loaded": true,
  "device": "cpu" # or "cuda" if GPU is available
}

Predict Sentiment

Endpoint: POST /predict

Description: Predicts the sentiment (positive or negative) of a given text.

Request Body: JSON object with a text field.

{
  "text": "This movie was absolutely fantastic! I loved every moment of it."
}

Example curl command (Positive):

curl -X POST -H "Content-Type: application/json" \
-d '{"text": "This movie was absolutely fantastic! I loved every moment of it."}' \
http://localhost:5000/predict

Expected Response (Positive):

{
  "text": "This movie was absolutely fantastic! I loved every moment of it.",
  "sentiment": "positive",
  "confidence": 0.9987
}

Example curl command (Negative):

curl -X POST -H "Content-Type: application/json" \
-d '{"text": "What a terrible film. I wasted my money and time."}' \
http://localhost:5000/predict

Expected Response (Negative):

{
  "text": "What a terrible film. I wasted my money and time.",
  "sentiment": "negative",
  "confidence": 0.9975
}

📈 Future Enhancements
Dedicated Database Connection Module: Implement connections/postgres_connection.py for reusable database connection logic.

Advanced Data Preprocessing: Incorporate more sophisticated text cleaning and feature engineering.

Model Versioning & Tracking: Integrate tools like MLflow to track experiments, manage model versions, and deploy models more formally.

CI/CD Pipeline: Automate the build, test, and deployment process using GitHub Actions, GitLab CI/CD, or Jenkins.

Scalable Deployment: Deploy the Docker image to cloud platforms (AWS ECS/EKS, Google Cloud Run/GKE, Azure Container Instances/AKS) for production-grade scaling.

Asynchronous Processing: For very high throughput, consider message queues (e.g., RabbitMQ, Kafka) for prediction requests.

User Interface: Build a simple web UI to interact with the API.

Feel free to explore, modify, and extend this project. Contributions are welcome!