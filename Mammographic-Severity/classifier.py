# Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('mammograph_dataset.csv')
df = pd.DataFrame(df)
df.head()

# Scale x values
ss = StandardScaler()
for col in df.columns:
  if col != 'Severity':
    df[col] = ss.fit_transform(df[[col]])

# Assign x and y values
y = list(df.pop('Severity'))
x = []
for rows in df.values.tolist():
  row = []
  for element in rows:
    row.append(float(element))
  x.append(row)

# Balance dataset (make sure there are an even representation of instances with label 1 and label 0)
smote = SMOTE()
x, y = smote.fit_resample(x, y)

# Divide the x and y values into two sets: train, and test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)

# Get input shape
input_shape = len(x[0])

# Create Adam optimizer
opt = Adam(learning_rate = 0.01)

# Create model
model = Sequential()

# Add an initial batch norm layer so that all the values are in a reasonable range for the network to process
model.add(BatchNormalization())
model.add(Dense(6, activation = 'relu', input_shape = [input_shape])) # Input layer

# Hidden layers
model.add(BatchNormalization())
model.add(Dense(4, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))

# Output layer
model.add(Dense(1, activation = 'sigmoid')) # Sigmoid because of binary classification

# Compile model
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Train model and store training history
epochs = 100
batch_size = 32
history = model.fit(x, y, epochs = epochs, validation_data = (x_test, y_test), batch_size = batch_size) # To add callbacks add 'callbacks = [early_stopping]'

# Visualize loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epoch_list = [i for i in range(epochs)]

plt.plot(epoch_list, loss, label = 'Loss')
plt.plot(epoch_list, val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize accuracy and validation accuracy
accuracy = history_dict['binary_accuracy']
val_accuracy = history_dict['val_binary_accuracy']

plt.plot(epoch_list, accuracy, label = 'Training Accuracy')
plt.plot(epoch_list, val_accuracy, label =' Validation Accuracy')
plt.title('Validation and Training Accuracy Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Prediction vs. actual value (change the index to view a different input and output set)
index = 0
prediction = np.argmax(model.predict([x_test[index]]))
print(f"\nModel's Prediction on a Sample Input: {prediction}")
print(f"Actual Label on the Same Input: {y_test[index]}")

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 0) # Change verbose to 1 or 2 for more information
print(f'\nTest accuracy: {test_acc * 100}%')

# View performance metrics
predict = model.predict(x_test)
predictions = [1.0 if j > 0.5 else 0 for j in predict] # Adjust values for classification report
print("\n", classification_report(y_test, predictions))
