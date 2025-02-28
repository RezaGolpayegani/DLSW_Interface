import ImageTools

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import ELU
from keras.utils.np_utils import to_categorical
import keras
import sys
import time

import numpy as np
import pickle

def network(img_size):
    # CNN based off of https://github.com/commaai/research/blob/master/train_steering_model.py
    net = Sequential()

    net.add(Conv2D(8, 5, 5, border_mode='valid', input_shape=(img_size, img_size, 1)))	
    net.add(Dropout(0.5))	
    net.add(Activation('relu'))
    net.add(Conv2D(16, 3, 3))
    net.add(Dropout(0.5))
    net.add(Activation('relu'))
    net.add(Conv2D(32, 3, 3))
    net.add(Dropout(0.5))
    net.add(Activation('relu'))
    net.add(Flatten())
    net.add(Dense(240))
    net.add(Activation('relu'))
    net.add(Dense(120))
    net.add(Dense(2))
    net.add(Activation('softmax'))
    return net

def main():
    img_size = 64
    classSize = 1000
    num_epochs = 15 

    # Loading Data
    print("\nImporting data..")
    food_files = ImageTools.parseImagePaths('./img/food/')
    sandwich_files = ImageTools.parseImagePaths('./img/sandwich/')
    print("\t..done.\n")

    print("\nAssigning Labels, Generating more images via transformation..")
    print("\tParsing/Labeling foods (sandwiches exclusive)..")
    food_x, food_y = ImageTools.expandClass(food_files, 0, classSize, img_size)
    print("\t\t..done.")
    print("\tParsing/Labeling sandwiches..")
    sandwich_x, sandwich_y = ImageTools.expandClass(sandwich_files, 1, classSize, img_size)
    print("\t\t..done.\n")

    # Arranging
    X = np.array(food_x + sandwich_x)
    y = np.array(food_y + sandwich_y)

    # Greyscaling and normalizing inputs to reduce features and improve comparability
    print("\nGreyscaling and Normalizing Images..")
    X = ImageTools.greyscaleImgs(X)
    X = ImageTools.normalizeImgs(X)
    print("\t..done.\n")
    y = to_categorical(y)

    # Train n' test:
    print("\nSplitting data into training and testing..")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=np.random.randint(0, 100))
    print("\t..done.\n")
    
    start_time = time.clock()

    print("\tCalling model..")
    model = network(img_size) # Calling of CNN
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    print("\t..done.\n")
    
    print("\nTraining Model..")
    print(X_train.shape)
    print(y_train.shape)
    model.fit(X_train, y_train, nb_epoch=num_epochs, validation_split=0.1,
              callbacks= [
                  #keras.callbacks.TerminateOnNaN(),
                  #keras.callbacks.EarlyStopping(monitor='loss', patience=1),
                  #keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
                  keras.callbacks.backpropagation(X_train, y_train,len(model.layers),32, start_time)
                  ])
    print("\t..done.\n")
    end_time = time.clock()
    print("time",(end_time- start_time))

    sys.exit(1)

    # Saving model
    print("\nPickling and saving model as 'model.pkl'...")
    modelsave = open('model.pkl', 'wb')
    pickle.dump(model, modelsave)
    modelsave.close()
    print("\t..done.\n")

main()
