from keras.models import Sequential
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers import Dropout, SpatialDropout1D, Flatten, Dense, Input, concatenate, Merge
from keras.regularizers import L1L2


def glove_model(input_dim, embedding_matrix, embedding_dim, input_length):
    print('Build model with GloVe embeddings...')
    model = Sequential()
    model.add(Embedding(input_dim,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=input_length,
                        trainable=True,
                        embeddings_regularizer=L1L2(l2=0.01)))
    model.add(SpatialDropout1D(0.2))
    model.add(Convolution1D(filters=512, kernel_size=3, padding='valid', activation='relu'))
    # model.add(Convolution1D(filters=64, kernel_size=3, padding='valid', strides=1, activation='relu'))
    model.add(MaxPooling1D(3))
    #model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    # model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(6, activation='softmax'))
    return model


def glove_model_trv(trv_size=50):
    print('Build model with GloVe embeddings...')
    model = Sequential()
    # model.add(Convolution1D(input_shape=(trv_size, 1), filters=512, kernel_size=3,
    #                         padding='valid', activation='relu'))
    # model.add(MaxPooling1D(3))
    model.add(Convolution1D(input_shape=(trv_size, 1), filters=512, kernel_size=3,
                            padding='valid', activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Convolution1D(filters=512, kernel_size=3,
                             padding='valid', activation='relu'))
    model.add(Convolution1D(filters=256, kernel_size=2,
                            padding='valid', activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(6, activation='softmax'))
    return model


def glove_model_lstm(trv_size=50):
    print('Build model with GloVe embeddings...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(trv_size, 1), dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(6, activation='softmax'))
    return model


def glove_model_multikernel(input_dim, embedding_matrix, embedding_dim, input_length):
    print('Build model with GloVe embeddings...')
    main_input = Input(shape=(input_length,), dtype='float32', name='main_input')
    embedding = Embedding(input_dim,
                          embedding_dim,
                          weights=[embedding_matrix],
                          input_length=input_length,
                          trainable=False)(main_input)
                          #trainable=True,
                          #embeddings_regularizer=L1L2(l2=0.01))(main_input)
    embedding = SpatialDropout1D(0.3)(embedding)
    conv_kernel_2 = Convolution1D(filters=128, kernel_size=2,
                                  padding='same', activation='relu')(embedding)
    conv_kernel_2 = MaxPooling1D(2)(conv_kernel_2)
    conv_kernel_3 = Convolution1D(filters=256, kernel_size=3,
                                  padding='same', activation='relu')(embedding)
    conv_kernel_3 = MaxPooling1D(2)(conv_kernel_3)
    conv_kernel_4 = Convolution1D(filters=64, kernel_size=4,
                                  padding='same', activation='relu')(embedding)
    conv_kernel_4 = MaxPooling1D(2)(conv_kernel_4)

    x = concatenate([conv_kernel_2, conv_kernel_3, conv_kernel_4])
    x = Flatten()(x)
    # x = Dropout(0.6)(x)
    # x = Dense(128, activation='relu', kernel_regularizer=L1L2(l2=0.01))(x)
    x = Dropout(0.6)(x)
    x = Dense(6, activation='softmax')(x)
    return Model(inputs=[main_input], outputs=[x])


def glove_model_trv_multikernel(trv_size=50):
    print('Build model with GloVe embeddings...')
    main_input = Input(shape=(trv_size, 1), dtype='float32', name='main_input')
    conv_kernel_2 = Convolution1D(filters=256, kernel_size=2,
                                  padding='same', activation='relu')(main_input)
    conv_kernel_2 = MaxPooling1D(2)(conv_kernel_2)
    conv_kernel_3 = Convolution1D(filters=512, kernel_size=3,
                                  padding='same', activation='relu')(main_input)
    conv_kernel_3 = MaxPooling1D(2)(conv_kernel_3)
    conv_kernel_4 = Convolution1D(filters=128, kernel_size=4,
                                  padding='same', activation='relu')(main_input)
    conv_kernel_4 = MaxPooling1D(2)(conv_kernel_4)

    x = concatenate([conv_kernel_2, conv_kernel_3, conv_kernel_4])
    #x = MaxPooling1D(4)(x)
    # model.add(Convolution1D(input_shape=(trv_size, 1), filters=512, kernel_size=3,
    #                         padding='valid', activation='relu'))
    # model.add(MaxPooling1D(3))
    # # model.add(Convolution1D(filters=512, kernel_size=3,
    # #                         padding='valid', activation='relu'))
    # # # model.add(Convolution1D(filters=256, kernel_size=2,
    # # #                         padding='valid', activation='relu'))
    # model.add(MaxPooling1D(3))
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.4)(x)
    x = Dense(6, activation='softmax')(x)
    return Model(inputs=[main_input], outputs=[x])


def glove_model_trv_lstm_cnn(trv_size=50):
    print('Build model with GloVe embeddings...')
    model = Sequential()
    model.add(Convolution1D(input_shape=(trv_size, 1), filters=512, kernel_size=3,
                            padding='valid', activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Dropout(0.4))
    model.add(LSTM(70))
    model.add(Dropout(0.4))
    model.add(Dense(6, activation='softmax'))
    return model


def glove_sentiment_model(input_dim, embedding_matrix, embedding_dim, input_length):
    print('Build model with GloVe embeddings...')
    model = Sequential()
    model.add(Embedding(input_dim,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=input_length,
                        trainable=True,
                        embeddings_regularizer=L1L2(l2=0.01)))
    model.add(SpatialDropout1D(0.2))
    model.add(Convolution1D(filters=512, kernel_size=3, padding='valid', activation='relu'))
    # model.add(Convolution1D(filters=64, kernel_size=3, padding='valid', strides=1, activation='relu'))
    model.add(MaxPooling1D(3))
    #model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    # model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))
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
    model.add(Convolution1D(filters=128, kernel_size=3, padding='valid', activation='relu'))
    model.add(Convolution1D(filters=64, kernel_size=3, padding='valid', activation='relu'))
    model.add(Convolution1D(filters=32, kernel_size=3, padding='valid', activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dropout(0.3))
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
