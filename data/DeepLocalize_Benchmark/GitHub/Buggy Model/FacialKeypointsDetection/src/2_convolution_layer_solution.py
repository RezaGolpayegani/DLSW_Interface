import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

dirname = os.path.dirname(os.path.realpath(__file__))
#place the .csv files one level up and within data/kaggle-facial-keypoint-detection folder.
FTRAIN = os.path.join(dirname, '../data/training.csv')
FTEST = os.path.join(dirname, '../data/test.csv')
MODEL_PATH = os.path.join(dirname,'../results/models/convoloution_layer.model')
LOG_PATH = os.path.join(dirname,'../results/logs/convolution_layer.csv')
MODEL_IMAGE = os.path.join(dirname,'../results/models/convolution_layer.png')

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def load2d(test=False, cols=None):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 96, 96, 1)
    return X, y

X, y = load2d()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))


# Create the keras model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras import optimizers
from keras.callbacks import CSVLogger
from keras.utils import plot_model
import keras.backend as K

def get_categorical_accuracy_keras(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))

#   Version - Convolutional layer.
model = Sequential()
#   input shape does not require batch size as none.
model.add(Conv2D(32, kernel_size=(3, 3), data_format='channels_last', input_shape=(96,96,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(30, activation='relu'))
model.summary()

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1)
#   Compile the model.  Minimize mean squared error  Maximize for accuracy
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=[get_categorical_accuracy_keras])


#   Add a logger to log the losses
csv_logger = CSVLogger(LOG_PATH, append=True, separator=';')
#   Fit the model with the data from make_blobs.  Make 100 cycles through the data.
#   Set verbose to 0 to supress progress messages 
history = model.fit(X, y, epochs=1000, verbose=1, validation_split=.2, shuffle=True, callbacks=[csv_logger])
#   Get loss and accuracy on test data
eval_result = model.evaluate(X, y)
#   Print test accuracy
print("\n\nTest loss:", eval_result)
# Save fine tuned model
model.save(MODEL_PATH)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
breakpoint()

# Plot training & validation accuracy values
plt.plot(history.history['get_categorical_accuracy_keras'])
plt.plot(history.history['val_get_categorical_accuracy_keras'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#   To visualize the model
plot_model(model, to_file=MODEL_IMAGE, show_shapes=True)
breakpoint()