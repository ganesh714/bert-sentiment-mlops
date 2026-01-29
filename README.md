# BERT Sentiment Analysis MLOps

This project implements a complete sentiment analysis system using a pre-trained BERT model. It includes data preprocessing, model fine-tuning, a REST API, a web UI, and containerization using Docker.

## Project Structure

```
├── data/
│   ├── raw/         # Place initial dataset here
│   └── processed/   # Store cleaned data here
├── model_output/    # Store fine-tuned model artifacts
├── results/         # Store evaluation metrics and predictions
├── scripts/         # For preprocessing, training, and prediction scripts
│   ├── preprocess.py
│   ├── train.py
│   └── batch_predict.py
├── src/
│   ├── api.py       # FastAPI application
│   └── ui.py        # Streamlit application
├── tests/           # API tests
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd bert-sentiment-mlops
    ```

2.  **Environment Setup:**
    Copy `.env.example` to `.env` and fill in the values.
    ```bash
    cp .env.example .env
    ```

3.  **Install Dependencies (Local Development):**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preprocessing
Run the preprocessing script to download and prepare the IMDB dataset.
```bash
python scripts/preprocess.py
```
This will create `data/processed/train.csv` and `data/processed/test.csv`.

### 2. Model Training
Run the training script to fine-tune the BERT model.
```bash
python scripts/train.py
```
**Note:** To run a quick debug training (for verification), use the `--debug` flag:
```bash
python scripts/train.py --debug
```
This will save the model to `model_output/` and metrics to `results/`.

### 3. Containerization
Build and run the services using Docker Compose.
```bash
docker-compose up --build
```
This will start:
-   **API:** http://localhost:8000 (Health check: `/health`, Docs: `/docs`)
-   **UI:** http://localhost:8501

### 4. Batch Prediction
Run the batch prediction script on a CSV file.
```bash
python scripts/batch_predict.py --input-file data/unseen/predict_data.csv --output-file results/predictions.csv
```

## detailed API Usage

### Health Check
**Endpoint:** `GET /health`
**Response:** `{"status": "ok"}`

### Predict Sentiment
**Endpoint:** `POST /predict`
**Body:** `{"text": "I loved this movie!"}`
**Response:** `{"sentiment": "positive", "confidence": 0.99}`

## Model Choice
We use `bert-base-uncased` from Hugging Face. BERT provides state-of-the-art results for text classification tasks. We use a standard sequence classification head on top of the pre-trained BERT model.

## Troubleshooting
-   **Model not found:** Ensure you have run `scripts/train.py` before building the Docker image for the API. The API container requires the `model_output/` directory to be populated.
-   **OOM Errors:** If running locally with `train.py`, try reducing the `batch_size` in the script.
