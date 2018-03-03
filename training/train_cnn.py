import argparse
import pickle
import time
import os

import numpy
import pandas
import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import CSVLogger
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from models.networks.cnn import glove_model, gloveless_model, glove_model_layered
# fix random seed for reproducibility
SEED = 7

MAX_FEATURES = 50000

# Maximal count of words in a sentence
MAX_WORDS = 37

# Map each label to integer value
EMOTION_LABELS_MAP = {'love': 0, 'happiness': 1, 'enthusiasm': 1, 'fun': 1, 'relief': 1, 'neutral': 2,
                      'surprise': 6, 'empty': 2, 'boredom': 2, 'worry': 3, 'sadness': 4, 'anger': 5,
                      'hate': 5}
# Map integers to labels (reverse of EMOTION_LABELS_MAP)
EMOTION_VALUES_MAP = {value: emotion for emotion, value in EMOTION_LABELS_MAP.items()}


def get_emotion_from_categorical(categorical):
    for i in range(len(categorical)):
        if categorical[i] == 1:
            return i


def top_2_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true=y_true, y_pred=y_pred, k=2)


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN network for emotion analysis.')
    parser.add_argument('-d', '--dataset-path', required=True,
                        type=str, help='Path to dataset CSV file.')
    parser.add_argument('-m', '--model-path', default='ea_cnn.h5',
                        type=str, help='Path to a file with trained emotion analysis model.')
    parser.add_argument('-t', '--tokenizer-path', default='tokenizer.pkl',
                        type=str, help='Path where tokenizer used during training will be saved.')
    parser.add_argument('--glove-embeddings-path', default=None,
                        type=str, help='Path to GloVE embeddings file.')
    parser.add_argument('--glove-embeddings-dim', default=None,
                        type=int, help='GloVe embedding size.')
    parser.add_argument('-e', '--epochs', default=None,
                        type=int, help='Epoch count.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    numpy.random.seed(SEED)

    USE_GLOVE = True if args.glove_embeddings_path and args.glove_embeddings_dim else False

    dataset = pandas.read_csv(args.dataset_path)

    tweets = dataset['content']
    sentiment = dataset['sentiment']

    # Print count of tweets from each class
    for emotion in EMOTION_LABELS_MAP.keys():
        emotion_tweets = [t[1] for t, s in zip(tweets.iteritems(), sentiment.iteritems()) if s[1] == emotion]
        print('Count of tweets with class {}: {}'.format(emotion, len(emotion_tweets)))

    # Preprocessing
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)

    # Dump fitted tokenizer
    with open(args.tokenizer_path, 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    preprocessed_texts = tokenizer.texts_to_sequences(tweets)
    word_index = tokenizer.word_index

    categorical_sentiment = [EMOTION_LABELS_MAP[sentiment_label] for sentiment_label in sentiment]
    categorical_sentiment = to_categorical(categorical_sentiment)

    # Padding
    preprocessed_texts = sequence.pad_sequences(preprocessed_texts, maxlen=MAX_WORDS)

    # Prepare model
    if USE_GLOVE:
        embeddings_index = {}
        with open(os.path.join(args.glove_embeddings_path), 'r', encoding='utf-8') as glove_file:
            for line in glove_file:
                values = line.split()
                word = values[0]
                coefs = numpy.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        lost_words_count = 0
        found_words_count = 0
        embedding_matrix = numpy.zeros((len(word_index) + 1, args.glove_embeddings_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
                found_words_count += 1
            else:
                lost_words_count += 1

        print('Found {} word vectors.'.format(len(embeddings_index)))
        print('GloVe representations not found for {} words.'.format(lost_words_count))
        print('GloVe representations found for {} words.'.format(found_words_count))
        model = glove_model(input_dim=len(word_index) + 1,
                            embedding_matrix=embedding_matrix,
                            embedding_dim=args.glove_embeddings_dim,
                            input_length=MAX_WORDS
                            )
    else:
        model = gloveless_model(input_dim=MAX_FEATURES, input_length=MAX_WORDS)

    # Split train/test data
    preprocessed_texts = numpy.array(preprocessed_texts)
    x_train, x_test, y_train, y_test = train_test_split(preprocessed_texts, categorical_sentiment,
                                                        test_size=0.25, random_state=SEED)

    print('Build model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[categorical_accuracy,
                           top_2_categorical_accuracy,
                           precision,
                           recall])
    print(model.summary())

    model.fit(x=x_train, y=y_train,
              validation_data=(x_test, y_test),
              epochs=args.epochs,
              batch_size=256,
              verbose=2,
              callbacks=[CSVLogger('training_{time}.csv'.format(time=time.time()))])

    # Evaluate model (on test data)
    predictions = model.predict(x_test, verbose=0)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true=[numpy.argmax(v) for v in y_test],
                                   y_pred=[numpy.argmax(pred) for pred in predictions])
    print(conf_matrix)

    model.save(args.model_path)
