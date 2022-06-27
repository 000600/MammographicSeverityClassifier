# Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier


# Initialize lists to store respective performances of each model
test_accuracy = []
train_accuracy = []

# Load dataset
df = pd.read_csv('MammographicDataset.csv')
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

# Divide the x and y values into three sets: train, test, and validation
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
history = model.fit(x, y, epochs = epochs, validation_data = (x_test, y_test), batch_size = batch_size, verbose = 0) # To add callbacks add 'callbacks = [early_stopping]'

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 0) # Change verbose to 1 or 2 for more information
train_loss, train_acc = model.evaluate(x_train, y_train, verbose = 0)
test_accuracy.append(float(test_acc))
train_accuracy.append(float(train_acc))

print(f'\nTesting Accuracy: {test_acc * 100}%')
print(f'Training Accuracy: {train_acc * 100}%')

# View performance metrics
predict = model.predict(x_test)
predictions = [1.0 if j > 0.5 else 0 for j in predict] # Adjust values for classification report
print("\n", classification_report(y_test, predictions))

## Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

# Get predictions for performance metrics
dtc_pred = dtc.predict(x_test)

# Add test and train accuracies to appropriate lists
dtc_acc = dtc.score(x_test, y_test)
test_accuracy.append(dtc_acc)
dtc_train_acc = dtc.score(x_train, y_train)
train_accuracy.append(dtc_train_acc)

print("Decision Tree Classifier")
print("========================")
print("Testing Accuracy :", dtc_acc)
print("Training Accuracy :", dtc_train_acc)

# Evaluate model
dtc_cr = classification_report(dtc_pred, y_test)
print(dtc_cr)

## Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

# Get predictions for performance metrics
logreg_pred = logreg.predict(x_test)

# Add test and train accuracies to appropriate lists
logreg_acc = logreg.score(x_test, y_test)
test_accuracy.append(logreg_acc)
logreg_train_acc = logreg.score(x_train, y_train)
train_accuracy.append(logreg_train_acc)

print("\nLogistic Regression")
print("===================")
print("Testing Accuracy :", logreg_acc)
print("Training Accuracy :", logreg_train_acc)

# Evaluate model
logreg_cr = classification_report(logreg_pred, y_test)
print(logreg_cr)

## Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

# Get predictions for performance metrics
rfc_pred = rfc.predict(x_test)

# Add test and train accuracies to appropriate lists
rfc_acc = rfc.score(x_test, y_test)
test_accuracy.append(rfc_acc) 
rfc_train_acc = rfc.score(x_train, y_train)
train_accuracy.append(rfc_train_acc)

print("\nRandom Forest Classifier")
print("=========================")
print("Testing Accuracy :", rfc_acc)
print("Training Accuracy :", rfc_train_acc)

# Evaluate model
rfc_cr = classification_report(rfc_pred, y_test)
print(rfc_cr)

## SVM Classifier
svc = SVC(C = 1.0, kernel = 'linear')
svc.fit(x_train, y_train)

# Get predictions for performance metrics
svc_pred = svc.predict(x_test)

# Add test and train accuracies to appropriate lists
svc_acc = svc.score(x_test, y_test)
test_accuracy.append(svc_acc)
svc_train_acc = svc.score(x_train, y_train)
train_accuracy.append(svc_train_acc)

print("\nSVM Classifier")
print("==============")
print("Testing Accuracy :", svc_acc)
print("Training Accuracy :", svc_train_acc)

# Evaluate model
svc_cr = classification_report(svc_pred, y_test)
print(svc_cr)

## KNN Classifier
knn = neighbors.KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train, y_train)

# Get predictions for performance metrics
knn_pred = knn.predict(x_test)

# Add test and train accuracies to appropriate lists
knn_acc = knn.score(x_test, y_test)
test_accuracy.append(knn_acc)
knn_train_acc = knn.score(x_train, y_train)
train_accuracy.append(knn_train_acc)

print("\nK-Nearest Neigjbor Classifier")
print("=============================")
print("Testing Accuracy :", knn_acc)
print("Training Accuracy :", knn_train_acc)

# Evaluate model
knn_cr = classification_report(knn_pred, y_test)
print(knn_cr)

n_groups = 6
objects = ('Neural Network', 'Decision Tree', 'Logistic Regression', 'Random Forest', 'SVM', 'KNN')

# Create plot
plt.figure(figsize = (10, 7))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

bar1 = plt.bar(index, test_accuracy, bar_width, alpha = opacity, color = 'blue', label = 'Test Accuracy')
bar2 = plt.bar(index + bar_width, train_accuracy, bar_width, alpha = opacity, color = 'orange', label = 'Train Accuracy')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison Across Models')
plt.xticks(index + bar_width, objects)
plt.legend()

plt.tight_layout()
plt.show()
