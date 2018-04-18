import argparse
import pickle
import time
import os

from imblearn.over_sampling import smote
import numpy
import pandas
import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import CSVLogger, EarlyStopping
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from models.networks.cnn import glove_model, gloveless_model, glove_model_layered, glove_model_trv
# fix random seed for reproducibility
SEED = 7

MAX_FEATURES = 50000

# Maximal count of words in a sentence
MAX_WORDS = 37

# Map each label to integer value
EMOTION_LABELS_MAP = {'love': 0, 'happiness': 1, 'enthusiasm': 1, 'fun': 1, 'relief': 1, 'neutral': 2,
                      'surprise': 2, 'empty': 2, 'boredom': 2, 'worry': 3, 'sadness': 4, 'anger': 5,
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


def create_glove_embedding_index(glove_embeddings_file_path):
    embeddings_index = {}
    with open(os.path.join(glove_embeddings_file_path), 'r', encoding='utf-8') as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = numpy.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    return embeddings_index


def create_text_representation_vectors(texts, word_index, embeddings_index,
                                       glove_embeddings_dim=300):
    lost_words_count = 0
    found_words_count = 0
    embedding_matrix = numpy.zeros((len(word_index) + 1, glove_embeddings_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            found_words_count += 1
        else:
            lost_words_count += 1
            # print('Lost word: {}'.format(word))

    # print('Found {} word vectors.'.format(len(embeddings_index)))
    # print('GloVe representations not found for {} words.'.format(lost_words_count))
    # print('GloVe representations found for {} words.'.format(found_words_count))

    # TRV - text representation vectors
    # Compute TRVs as linear combination of word embedding vectors
    text_representation_vectors = []
    for text in texts:
        trv = numpy.sum([embedding_matrix[word_id] for word_id in text], axis=0)
        text_representation_vectors.append(trv)

    return text_representation_vectors


def show_pca_and_lda_plots(x_res, y_res):
    pca = PCA(n_components=4)
    X_r = pca.fit(x_res).transform(x_res)
    lda = LinearDiscriminantAnalysis(n_components=4)
    X_r2 = lda.fit(x_res, y_res).transform(x_res)
    colors = ['navy', 'turquoise', 'darkorange', 'red', 'yellow', 'green', 'purple']
    lw = 2
    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5], ['love', 'happiness',
                                                         'neutral', 'worry',
                                                         'sadness', 'hate']):
        plt.scatter(X_r[y_res == i, 0], X_r[y_res == i, 1], color=color, alpha=.3, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of dataset after SMOTE.')
    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5], ['love', 'happiness',
                                                         'neutral', 'worry',
                                                         'sadness', 'hate']):
        plt.scatter(X_r2[y_res == i, 0], X_r2[y_res == i, 1], alpha=.5, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of dataset after SMOTE.')
    plt.show()

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
    tokenizer = Tokenizer(filters='!"#$%&*+,-.<=>?@[\\]^_`{}~\t\n')
    tokenizer.fit_on_texts(tweets)

    # Dump fitted tokenizer
    with open(args.tokenizer_path, 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    preprocessed_texts = tokenizer.texts_to_sequences(tweets)

    text_labels = [EMOTION_LABELS_MAP[sentiment_label] for sentiment_label in sentiment]
    text_labels = to_categorical(text_labels)

    # Padding
    preprocessed_texts = sequence.pad_sequences(preprocessed_texts, maxlen=MAX_WORDS)

    # Prepare model
    embedding_index = create_glove_embedding_index(args.glove_embeddings_path)
    # Construct text representation vectors
    trvs = create_text_representation_vectors(texts=preprocessed_texts,
                                              word_index=tokenizer.word_index,
                                              embeddings_index=embedding_index,
                                              glove_embeddings_dim=args.glove_embeddings_dim)
    # SMOTE
    sm = smote.SMOTE()
    smote_labels = [numpy.argmax(label) for label in text_labels]
    x_res, y_res = sm.fit_sample(trvs, smote_labels)

    # Expand x_res to 3 dimensions
    x_res = numpy.expand_dims(x_res, axis=2)
    # Convert text_labels back to ocategorical
    y_res = to_categorical(y_res)
    # Split train/test data
    x_train, x_test, y_train, y_test = train_test_split(x_res, y_res,
                                                        test_size=0.20, random_state=SEED)
    print(x_train[0])
    print(y_train[0])
    model = glove_model_trv(trv_size=args.glove_embeddings_dim)

    print('Build model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[categorical_accuracy,
                           top_2_categorical_accuracy])
    print(model.summary())

    # plot_model(model, to_file='model.png')

    model.fit(x=x_train, y=y_train,
              validation_data=(x_test, y_test),
              epochs=args.epochs,
              batch_size=256,
              verbose=1,
              callbacks=[CSVLogger('training_{time}.csv'.format(time=time.time())),
                         EarlyStopping(monitor='val_loss', min_delta=0.001, patience=8, verbose=1,
                                       mode='auto')])

    # Evaluate model (on test data)
    predictions = model.predict(x_test, verbose=0)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true=[numpy.argmax(v) for v in y_test],
                                   y_pred=[numpy.argmax(pred) for pred in predictions])
    print(conf_matrix)

    model.save(args.model_path)
