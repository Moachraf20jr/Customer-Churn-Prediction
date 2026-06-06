import sys, json, os, numpy as np, pandas as pd, joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    encoders = joblib.load(os.path.join(MODELS_DIR, 'encoders.pkl'))
    metadata = joblib.load(os.path.join(MODELS_DIR, 'metadata.pkl'))
    return model, scaler, encoders, metadata

model, scaler, encoders, metadata = load_artifacts()
expected_features = metadata['feature_names']

NUMERIC_FIELDS = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']

def predict(data_list):
    if isinstance(data_list, dict):
        data_list = [data_list]
    df = pd.DataFrame(data_list)

    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    for col in df.select_dtypes(include=['object', 'str']).columns.tolist():
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col])
            except ValueError:
                unseen = ~df[col].isin(encoders[col].classes_)
                if unseen.any():
                    df.loc[unseen, col] = encoders[col].classes_[0]
                df[col] = encoders[col].transform(df[col])

    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])

    if 'Usage Frequency' in df.columns and 'Support Calls' in df.columns:
        df['Usage_Support_Ratio'] = df['Usage Frequency'] / (df['Support Calls'] + 1)
    if 'Total Spend' in df.columns and 'Tenure' in df.columns:
        df['Spend_Per_Month'] = df['Total Spend'] / (df['Tenure'] + 1)
    if 'Last Interaction' in df.columns:
        df['Days_Since_Interaction'] = df['Last Interaction']

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]
    scaled = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    preds = model.predict(scaled)
    probs = model.predict_proba(scaled)[:, 1]

    results = []
    for i, (p, prob) in enumerate(zip(preds, probs)):
        results.append({
            'prediction': 'Churn' if p == 1 else 'No Churn',
            'probability': round(float(prob), 4),
            'risk_level': 'High' if prob >= 0.7 else ('Medium' if prob >= 0.3 else 'Low')
        })
    return results

if __name__ == '__main__':
    try:
        input_data = json.loads(sys.stdin.read())
        output = predict(input_data)
        print(json.dumps(output))
    except json.JSONDecodeError as e:
        print(json.dumps({'error': 'Invalid JSON input: ' + str(e)}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)
