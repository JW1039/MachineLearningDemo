import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential

data = pd.read_csv("data/purchase_data.csv")
print(data)
model = Sequential()
