from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.python.keras.backend import conv2d, dtype, mean, std
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers.core import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold
import pretty_errors
from tensorflow.keras.layers import Conv2D
import random
from tensorflow.keras.callbacks import ModelCheckpoint
import os

def load_dataset():
    (trainX,trainY), (testX,testY) = mnist.load_data()

    #reshape data 
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    # encode each class (number) as a numerical value

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    
    return trainX,trainY,testX,testY

def prep_pixels(train,test):
    
    # integers to floats 
    train_norm = train.astype('int32')
    test_norm = test.astype('int32')
    # normalize pixel values to 0-1
    train_norm = train_norm/255
    test_norm = test_norm/255
    
    # return normalized values 
    return train_norm, test_norm

def define_model():
    model = Sequential()
    model.add(Conv2D(32,(3,3), activation='relu',kernel_initializer='he_uniform', input_shape=(28,28,1)))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    
    # compile model 
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model

def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    
    # enumurate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define model 
        model = define_model()
        # select rows for training and testing 
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        
        #fit model
        history = model.fit(trainX, trainY, epochs=2, batch_size=32, validation_data=(testX, testY),callbacks=[cp_callback],verbose=1)
        
        # evaluate model 
        _, acc = model.evaluate(testX, testY, verbose=1)
        print(f'> {acc*100.0}%')
        
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories, model

def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss 
        plt.subplot(2,1,1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2,1,2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()
    

def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves 
    summarize_diagnostics(histories)

def try_one():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    
    # evaluate model
    scores, histories, model = evaluate_model(trainX,trainY)
    image_index = random.randint(0,6000)
    plt.imshow(testX[image_index].reshape(28,28),cmap='Greys')
    print((testY[image_index]))
    pred = model.predict(testX[image_index].reshape(1,28,28,1))
    print(pred.argmax())
    plt.show()

    
if __name__ == '__main__':
    # run_test_harness()
    
    # code for first run of model
    '''cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    try_one()'''
    
    # checkpoint paths
    checkpoint_path = 'training_1/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # loading model
    trainX, trainY, testX, testY = load_dataset()
    model = define_model()
    model.load_weights(checkpoint_path)
    
    # evaluating model accuracy
    loss, acc = model.evaluate(testX,testY,verbose=10)
    print(f'Accuracy is {acc*100}%')
    
    # showing examples of guessing
    image_index = random.randint(0,6000)
    plt.subplot(2,2,1)
    plt.imshow(testX[image_index].reshape(28,28),cmap='Greys')
    plt.title('Test Image')
    pred = model.predict(testX[image_index].reshape(1,28,28,1))
    plt.subplot(2,2,2)
    plt.text(.5,.5,str(pred.argmax()),fontsize=44)
    plt.title('Model guess')
    plt.show()
