
from keras.utils import np_utils
from ser.__init__ import Model
from ser.dnn import LSTM, CNN

from ser.utilities import get_data, class_labels

dataset_path = 'dataset'


def dnn_speech():
    x_train, x_test, y_train, y_test = get_data(dataset_path=dataset_path, flatten=False)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    print( 'Starting LSTM')
    model = LSTM(input_shape=x_train[0].shape, num_classes=len(class_labels))
    model.train(x_train, y_train, x_test, y_test)
    model.evaluate(x_test, y_test)
    print( 'LSTM Done\n Starting CNN')
    in_shape = x_train[0].shape
    x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
    model = CNN(input_shape=x_train[0].shape, num_classes=len(class_labels))
    model.train(x_train, y_train, x_test, y_test)
    model.evaluate(x_test, y_test)
    print ('CNN Done')


if __name__ == "__main__":

    dnn_speech()

    
    


