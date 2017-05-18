import pickle
import time
import os

import numpy
import pandas
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import CSVLogger
from keras.metrics import top_k_categorical_accuracy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# fix random seed for reproducibility
SEED = 123
DATASET_PATH = 'text_emotion.csv'
max_features = 50000

emotion_labels = {'love': 0, 'happiness': 1, 'enthusiasm': 1, 'fun': 1, 'relief': 1, 'neutral': 2,
                  'surprise': 2, 'empty': 2, 'boredom': 2, 'worry': 3, 'sadness': 4, 'anger': 5,
                  'hate': 5}

emotion_indices = {0: 'love', 1: 'happiness', 2: 'neutral', 3: 'worry', 4: 'sadness', 5: 'hate'}

emotion_labels_array = ['love', 'happiness', 'neutral', 'worry', 'sadness', 'anger']


def gloveless_model():
    print('Build model without GloVe embeddings...')
    model = Sequential()
    model.add(Embedding(max_features, 8, input_length=MAX_WORDS, dropout=0.6))
    model.add(Convolution1D(filters=64, kernel_size=3, padding='valid', strides=1, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[top_2_categorical_accuracy])
    return model


def glove_model():
    print('Build model with GloVe embeddings...')
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_WORDS,
                        trainable=False,
                        dropout=0.4))
    model.add(Convolution1D(filters=256, kernel_size=3, padding='valid', strides=1, activation='relu'))
    # model.add(Convolution1D(filters=64, kernel_size=3, padding='valid', strides=1, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    # model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(6, activation='softmax'))
    return model


def get_emotion_from_categorical(categorical):
    for i in range(len(categorical)):
        if categorical[i] == 1:
            return i


def top_2_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true=y_true, y_pred=y_pred, k=2)


if __name__ == '__main__':
    numpy.random.seed(SEED)

    # load the dataset
    dataset = pandas.read_csv(DATASET_PATH)

    tweets = dataset['content']
    categorical_sentiment = dataset['sentiment']

    # Preprocessing
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)

    # Dump fitted tokenizer
    with open('tokenizer.bin', 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    preprocessed_texts = tokenizer.texts_to_sequences(tweets)
    word_index = tokenizer.word_index

    categorical_sentiment = [emotion_labels[e] for e in categorical_sentiment]
    categorical_sentiment = to_categorical(categorical_sentiment)

    # Padding
    MAX_WORDS = 37
    EMBEDDING_DIM = 200
    preprocessed_texts = sequence.pad_sequences(preprocessed_texts, maxlen=MAX_WORDS)

    # GloVe embeddings
    embeddings_index = {}
    with open(os.path.join('glove.6B.{}d.txt'.format(EMBEDDING_DIM)), 'r', encoding='utf-8') as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = numpy.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    lost_words_count = 0
    found_words_count = 0
    embedding_matrix = numpy.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            found_words_count += 1
        else:
            lost_words_count += 1

    print('Found %s word vectors.' % len(embeddings_index))
    print('GloVe representations not found for %s words.' % lost_words_count)
    print('GloVe representations found for %s words.' % found_words_count)

    # Split train/test data
    preprocessed_texts = numpy.array(preprocessed_texts)
    x_train, x_test, y_train, y_test = train_test_split(preprocessed_texts, categorical_sentiment,
                                                        test_size=0.3, random_state=SEED)

    print('Build model...')
    model = glove_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[top_2_categorical_accuracy])
    print(model.summary())

    model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=16, batch_size=256, verbose=2,
              callbacks=[CSVLogger('training_{time}.csv'.format(time=time.time()))])

    # Evaluate model (on test data)
    predictions = model.predict(x_test, verbose=0)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true=[numpy.argmax(v) for v in y_test],
                                   y_pred=[numpy.argmax(pred) for pred in predictions])
    print(conf_matrix)

    # print("Accuracy: %.2f%%" % (scores[1]*100))

    model.save('emotion_analysis_CNN_keras_2.h5')
