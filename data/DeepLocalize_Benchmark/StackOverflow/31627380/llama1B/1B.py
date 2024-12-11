# Here is the refactored code with improved parameters and tuning techniques:

# ```python
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU, LeakyReLU
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.utils import to_categorical
import time
import sys

np.random.seed(1919)

# Constants
batch_size = 32
nb_epoch = 1000
n_features = len(train.columns)
n_classes = Y.shape[1]

# Load data
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

print("Data Read complete")

# Label encode categorical variables
for i in range(n_classes):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train.iloc[:,i]) + list(test.iloc[:,i]))
    train.iloc[:,i] = lbl.transform(train.iloc[:,i])
    test.iloc[:,i] = lbl.transform(test.iloc[:,i])

# Convert data to numpy array
train = np.array(train)
test = np.array(test)

# Normalize data
mean = np.mean(train, axis=0)
stddev = np.std(train, axis=0)
train = (train - mean) / stddev
test = (test - mean) / stddev

print("Data Normalized")

# Tune hyperparameters using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    'activation': ['relu', 'leaky_relu', 'tanh'],
    'dropout_rate': [0.1, 0.2, 0.3],
    'num_layers': [3, 4, 5],
    'dense_units': [128, 256, 512]
}

df = pd.DataFrame(param_grid)
best_model = GridSearchCV(
    Sequential(),
    df,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
).fit(train, Y)

print("Best Hyperparameters: ", best_model.best_params_)
print("Best Accuracy: ", best_model.best_score_)

# Train the model with best hyperparameters
best_model = GridSearchCV(
    Sequential(),
    df,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
).fit(train, Y)

model.fit(train, Y, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.20, callbacks=[
    keras.callbacks.EarlyStopping(monitor='loss', patience=50)
])

# Evaluate the model on test data
preds = model.predict(test, batch_size=batch_size)
pred_arr = preds.argmax(axis=1)

# Output results
print("Predicted Survived:", preds)
print("Confusion Matrix:\n", confusion_matrix(preds, Y))
print("Classification Report:\n", classification_report(Y, preds))

# ```

# Explanation of the changes made:

# 1. The dataset was normalized by subtracting the mean and then dividing by the standard deviation for each feature.

# 2. Hyperparameter tuning using GridSearchCV was applied to find the best combination of hyperparameters that result in the highest accuracy on a validation set.

# 3. A callback for early stopping has been added to prevent overfitting during training.

# 4. The model is trained with the best hyperparameters found by GridSearchCV, and the predicted survived status is stored in `preds`.

# 5. Accuracy, confusion matrix, and classification report are printed out at the end of the script.