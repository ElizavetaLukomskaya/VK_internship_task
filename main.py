import pandas as pd
import numpy as np
import pickle
import pyarrow
import warnings
warnings.filterwarnings('ignore')

def load_model(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

    return loaded_model

def generate_val_features(df):
    values_array = np.array(df['values'].tolist(), dtype=object)

    return pd.DataFrame({
        'id': df['id'],
        'val_mean': [np.mean(val) for val in values_array],
        'val_std': [np.std(val) for val in values_array],
        'val_max': [np.max(val) for val in values_array],
        'val_min': [np.min(val) for val in values_array],
        'val_median': [np.median(val) for val in values_array],
        'val_25%': [np.percentile(val, 25) for val in values_array],
        'val_75%': [np.percentile(val, 75) for val in values_array],
    })

def preprocess_data(input_data):
    input_data = input_data.merge(generate_val_features(input_data), on='id', how='left')
    input_data = input_data.explode(['dates', 'values'])
    input_data['dates'] = pd.to_datetime(input_data['dates'], format='%Y-%m-%d')

    for feature in ['val_mean', 'val_std', 'val_max', 'val_min', 'val_median', 'val_25%', 'val_75%']:
        input_data[feature].fillna(input_data[feature].mean(), inplace=True)

    input_data['year'] = input_data['dates'].dt.year
    min_max_year = input_data.groupby('id').agg(min_date=('dates', 'min'), 
                                         max_date=('dates', 'max'), 
                                         min_year=('year', 'min'), 
                                         max_year=('year', 'max')).reset_index()
    
    input_data = input_data.merge(min_max_year, on='id', how='left')
    input_data['date_diff'] = input_data['max_date'].dt.to_period('M').view(dtype='int64') - input_data['min_date'].dt.to_period('M').view(dtype='int64')
    input_data = input_data.drop(columns=['dates', 'values', 'year', 'min_date', 'max_date'])
    input_data = input_data.drop_duplicates().reset_index().drop('index', axis=1)
    return input_data

def make_predictions(model, input_data):
    return model.predict_proba(input_data.drop('id', axis=1))[:, 1]

def save_results(input_data, predictions):
    submission = pd.DataFrame({
    'id': input_data['id'],
    'score': predictions})

    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":

    model = load_model('model.pkl')

    input_data = pd.read_parquet('test.parquet')

    input_data = preprocess_data(input_data)

    predictions = make_predictions(model, input_data)

    save_results(input_data, predictions)