import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
model = load_model('model.h5')

with open('onehot_encoder_geo.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scalar = pickle.load(file)

input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Female',
    'Age': 22,
    'Tenure': 3,
    'Salary': 60000
}
geo_encoded = label_encoder_geo.transform(
    [[input_data['Geography']]]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=label_encoder_geo.get_feature_names_out(['Geography'])
)
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
input_df = pd.DataFrame(
     [input_df.drop("Geography", axis=1), geo_encoded_df],
    axis=1
)
input_scaled = scalar.transform(input_df)

prediction = model.predict(input_scaled)

prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    print('The customer is likely to churn.')
else:
    print('The customer is not likely to churn.')
    #threshold=0.5
    # > 0.5 -> churn
    # <= 0.5 -> no churn