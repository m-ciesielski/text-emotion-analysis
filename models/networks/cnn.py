from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Dropout, SpatialDropout1D, Flatten, Dense


def glove_model(input_dim, embedding_matrix, embedding_dim, input_length):
    print('Build model with GloVe embeddings...')
    model = Sequential()
    model.add(Embedding(input_dim,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=input_length,
                        trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(Convolution1D(filters=512, kernel_size=3, padding='valid', activation='relu'))
    # model.add(Convolution1D(filters=64, kernel_size=3, padding='valid', strides=1, activation='relu'))
    model.add(MaxPooling1D(3))
    #model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    # model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(6, activation='softmax'))
    return model


def glove_model_layered(input_dim, embedding_matrix, embedding_dim, input_length):
    print('Build model with GloVe embeddings...')
    model = Sequential()
    model.add(Embedding(input_dim,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=input_length,
                        trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(Convolution1D(filters=256, kernel_size=3, padding='valid', activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Convolution1D(filters=64, kernel_size=3, padding='valid', activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))
    return model


def gloveless_model(input_dim, input_length):
    print('Build model without GloVe embeddings...')
    model = Sequential()
    model.add(Embedding(input_dim, 8, input_length=input_length))
    model.add(SpatialDropout1D(0.4))
    model.add(Convolution1D(filters=64, kernel_size=3, padding='valid', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(6, activation='softmax'))
    return model