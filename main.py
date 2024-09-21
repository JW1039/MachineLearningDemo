# import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout
from constants import *
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# Load dataset
df = pd.read_csv("data/purchase_data.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Normalize features
scaler = MinMaxScaler()
df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])

# Split the data into features (X) and target (y)
X = df[FEATURE_COLUMNS]

# Encode the target for multiclass classification
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COLUMN])

# One-hot encode the target for multiclass classification
y = to_categorical(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define input shape
input_shape = len(FEATURE_COLUMNS)

# Build the model
model = Sequential()

# First Dense layer
model.add(Dense(units=64, input_shape=(input_shape,), activation='relu'))

# Second Dense layer
model.add(Dense(units=32, activation='relu'))

# Dropout layer to prevent overfitting
model.add(Dropout(0.2))

# Output layer (number of units should match number of classes)
# Since y_train is one-hot encoded, the number of classes is the number of columns in y_train
model.add(Dense(units=y_train.shape[1], activation='softmax'))  # Multiclass output

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
model.evaluate(X_test, y_test)

# Function to predict next product category for a given customer_id
def predict_next_purchase(customer_id):
    # Extract the customer's data from the dataset
    customer_data = df[df['customer_id'] == customer_id][FEATURE_COLUMNS]
    
    if customer_data.empty:
        print(f"No data found for customer_id {customer_id}")
        return
    
    # Scale the customer's features (make sure the scaler is consistent with training)
    customer_data_scaled = scaler.transform(customer_data)

    # Make the prediction
    prediction = model.predict(customer_data_scaled)
    
    # Convert one-hot prediction back to label
    predicted_category = le.inverse_transform([np.argmax(prediction)])
    
    print(f"Predicted next product category for customer_id {customer_id}: {predicted_category[0]}")

# Example usage for a specific customer_id
for cid in TEST_VALUES:
    predict_next_purchase(cid)
