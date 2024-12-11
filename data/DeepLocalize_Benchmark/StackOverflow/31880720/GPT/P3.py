# There are a few issues with the provided deep learning code. I’ll break them down and provide fixes for each one:

# ### 1. **Incorrect Usage of `softmax` Activation**:
#    - **Problem**: `softmax` is used in the last layer for binary classification. However, `softmax` is typically used for multi-class classification. For binary classification, `sigmoid` is the appropriate activation function.
#    - **Fix**: Replace `softmax` with `sigmoid` in the last layer.

# ### 2. **Inappropriate Loss Function**:
#    - **Problem**: The loss function `binary_crossentropy` is correct, but it is paired with a `softmax` activation, which is incorrect. Since this is a binary classification problem, `sigmoid` should be paired with `binary_crossentropy`.
#    - **Fix**: This will be resolved once the activation function is fixed.

# ### 3. **Redundant Activation Layer**:
#    - **Problem**: You have redundant `Activation('relu')` calls after defining the activation in the `Dense` layers.
#    - **Fix**: Remove the extra `Activation('relu')` calls.

# ### 4. **Deprecated `time.clock()`**:
#    - **Problem**: `time.clock()` has been deprecated in Python 3.8+. Use `time.time()` instead for measuring the runtime.
#    - **Fix**: Replace `time.clock()` with `time.time()`.

# ### 5. **Batch Size Setting**:
#    - **Problem**: The batch size is set to 1, which might slow down training. You may want to use a higher batch size, such as 32.
#    - **Fix**: Optionally, adjust the batch size based on the system's resources.

# Here’s the corrected version of the code:

# ```python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:43:51 2019

@author: PC_Wardat
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import time

# Load dataset (example, you can ignore the mnist dataset if irrelevant)
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()
print(x_tr.shape)

# Define custom dataset (as per your example)
X_train = np.array([[1] * 128] * (10 ** 4) + [[0] * 128] * (10 ** 4))
X_test = np.array([[1] * 128] * (10 ** 2) + [[0] * 128] * (10 ** 2))

Y_train = np.array([True] * (10 ** 4) + [False] * (10 ** 4))
Y_test = np.array([True] * (10 ** 2) + [False] * (10 ** 2))

print(Y_train.shape)

# Cast data types
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
Y_train = Y_train.astype("bool")
Y_test = Y_test.astype("bool")

# Model parameters
batch_size = 32  # Batch size set to 32 instead of 1 for better performance
epochs = 3

# Build the model
model = Sequential()
model.add(Dense(units=50, activation='relu', input_dim=128))
model.add(Dropout(0.2))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))  # Changed from 'softmax' to 'sigmoid'

# Compile the model
rms = RMSprop()
model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])

# Measure the start time
start_time = time.time()

# Train the model
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test),
          callbacks=[keras.callbacks.TerminateOnNaN(),
                     keras.callbacks.EarlyStopping(monitor='loss', patience=1)])

# Measure the end time
end_time = time.time()

# Evaluate the model
score = model.evaluate(X_test, Y_test, verbose=1)

# Print runtime and results
print("Run time:", (end_time - start_time))
print('Test score:', score[0])
print('Test accuracy:', score[1])
# ```

### Key Fixes:
# 1. Changed the activation in the last layer from `softmax` to `sigmoid` for binary classification.
# 2. Removed redundant activation layers.
# 3. Updated `time.clock()` to `time.time()`.
# 4. Adjusted batch size to 32 for better training efficiency.

# This corrected version should improve the accuracy of the model by addressing the key issues, particularly the incorrect activation function.