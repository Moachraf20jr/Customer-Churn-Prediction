# Customer Churn Prediction

Predict customer churn using machine learning. Built with scikit-learn (Random Forest) and served via Node.js/Express dashboard.

## Quick Start

```bash
npm install
pip install -r requirements.txt
python main.py        # train model
node server.js        # start dashboard
```

## Structure

- `main.py` - ML pipeline (train, evaluate, save model)
- `server.js` - Express server
- `public/` - Dashboard frontend
- `predict_service.py` - Prediction API called by server
- `models/` - Trained model files
- `results/` - Generated plots and metrics

## API

- `GET /api/health` - Health check
- `GET /api/model-info` - Model metadata & metrics
- `POST /api/predict` - Predict churn (accepts array of customer objects)

## Model

Random Forest classifier trained on customer data. Features: Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend, Last Interaction.

Accuracy: 93.2% | F1: 94.2% | ROC-AUC: 95.2%
