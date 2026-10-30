# SpamShield: ML-Based Spam & Phishing Message Classifier

SpamShield classifies SMS/email text as **spam** or **ham** using a TF-IDF + scikit-learn pipeline.
It trains and compares three models, saves the best one, and serves predictions through a Flask REST API.

## Results (20% stratified hold-out, 5,169 de-duplicated messages)

| Model               | Accuracy | Precision | Recall | F1     |
|---------------------|----------|-----------|--------|--------|
| Naive Bayes         | 0.9691   | 1.0000    | 0.7557 | 0.8609 |
| Logistic Regression | 0.9739   | 0.9062    | 0.8855 | 0.8958 |
| Random Forest       | 0.9768   | 0.9908    | 0.8244 | 0.9000 |

The best model by F1 is saved automatically. A confusion matrix is written to `models/confusion_matrix.png`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py                  # downloads data, trains, saves model
python app.py                    # starts API on http://localhost:5000
python -m unittest test_app -v   # run tests
```

## API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "WINNER! Claim your free prize now"}'
```

Response: `{"label": "spam", "spam_probability": 0.87, "message": "..."}`

Other endpoints: `GET /health`, `POST /predict/batch` with `{"messages": [...]}`.

## Project structure

```
train.py       data loading, model comparison, evaluation, saving
app.py         Flask REST API
test_app.py    API tests
models/        saved model, metrics.json, confusion matrix
data/          dataset (downloaded on first run)
```

## Roadmap / looking for collaborators

- Add a transformer-based model (e.g. DistilBERT) and compare against the baselines
- Build a simple web UI for pasting messages
- Add cross-validation and threshold tuning to improve spam recall
- Dockerize and deploy (Render / Hugging Face Spaces)

Dataset: SMS Spam Collection (UCI).
