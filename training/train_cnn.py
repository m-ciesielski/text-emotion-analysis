import argparse
from collections import defaultdict
import pickle
import random
import time
import os

import re
from imblearn.over_sampling import smote, adasyn
import numpy
import pandas
import keras.backend as K
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import CSVLogger, EarlyStopping
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from models.networks.cnn import glove_model_trv, glove_model, glove_model_multikernel, glove_model_lstm, glove_model_layered
# fix random seed for reproducibility
SEED = 7

MAX_FEATURES = 50000

# Maximal count of words in a sentence
MAX_WORDS = 37

# Map each label to integer value
EMOTION_LABELS_MAP = {'love': 0, 'happiness': 1, 'enthusiasm': 1, 'fun': 1, 'relief': None, 'neutral': 2,
                      'surprise': None, 'empty': 2, 'boredom': 2, 'worry': 3, 'sadness': 4, 'anger': 5,
                      'hate': 5}
# Map integers to labels (reverse of EMOTION_LABELS_MAP)
EMOTION_VALUES_MAP = {value: emotion for emotion, value in EMOTION_LABELS_MAP.items() if value is not None}


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


def create_embedding_matrix(word_index, embeddings_index,
                            glove_embeddings_dim=300):
    print(glove_embeddings_dim)
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

    return embedding_matrix


def create_text_representation_vectors(texts, embedding_matrix):
    # TRV - text representation vectors
    # Compute TRVs as linear combination of word embedding vectors
    text_representation_vectors = []
    for text in texts:
        trv = numpy.sum([embedding_matrix[word_id] for word_id in text], axis=0)
        text_representation_vectors.append(trv)

    return text_representation_vectors


def show_pca_and_lda_plots(x, y, max_items=None):
    if max_items:
        x, y = x[:max_items], y[:max_items]

    pca = PCA(n_components=2)
    X_r = pca.fit(x).transform(x)
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(x, y).transform(x)
    # colors = ['navy', 'turquoise', 'darkorange', 'red', 'yellow', 'green', 'purple']
    colors = ['green', 'blue']
    lw = 2
    plt.figure()
    for color, i, target_name in zip(colors, [0, 1], ['love', 'happiness']):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.5, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of dataset after SMOTE.')
    plt.figure()
    for color, i, target_name in zip(colors, [0, 1], ['love', 'happiness']):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.5, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of dataset after SMOTE.')
    plt.show()


def show_tsne_plot(x, y, max_items=10000):
    if max_items:
        x, y = x[:max_items], y[:max_items]

    # Reduce initial dimensionality with PCA
    pca = PCA(n_components=50)
    X_r = pca.fit(x).transform(x)

    print('Cumulative explained variation for 50 principal components: {}'.format(
        numpy.sum(pca.explained_variance_ratio_)))

    tsne_start = time.time()
    X_embedded = TSNE(n_components=2).fit_transform(X_r)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - tsne_start))

    colors = ['deeppink', 'purple']
    # colors = ['navy', 'turquoise', 'darkorange', 'red', 'yellow', 'green', 'purple']
    for color, i, target_name in zip(colors, [0, 5], ['love', 'hate']):
        plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], alpha=.7, color=color,
                    label=target_name)

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('T-SNE of dataset after SMOTE.')
    plt.show()


def oversample_with_duplicates(x: list or numpy.ndarray, y: list or numpy.ndarray, ratio=1.0):
    # Calculate number of examples for each class

    oversampled_x = list(x)
    oversampled_y = list(y)

    class_sizes = defaultdict(int)
    for label in y:
        class_sizes[label] += 1

    target_size = max(class_size for class_size in class_sizes.values()) * ratio

    for class_label, class_size in class_sizes.items():
        class_items = [item for item, label in zip(x, y) if label == class_label]
        items_to_add = target_size - class_size
        items_to_add = items_to_add if items_to_add > 0 else 0
        print('Duplicating {} items for class {}.'.format(items_to_add,
                                                          class_label))
        while items_to_add > 0:
            item_to_duplicate = random.choice(class_items)
            oversampled_x.append(item_to_duplicate)
            oversampled_y.append(class_label)
            items_to_add -= 1

    return oversampled_x, oversampled_y


def random_undersampling(x: list or numpy.ndarray, y: list or numpy.ndarray, ratio=0.8):
    undersampled_x = list(x)
    undersampled_y = list(y)

    class_sizes = defaultdict(int)
    for label in y:
        class_sizes[label] += 1

    target_size = int(max(class_size for class_size in class_sizes.values()) * ratio)

    class_items_to_remove = defaultdict(int)
    for class_label, class_size in class_sizes.items():
        class_items_to_remove[class_label] = class_size - target_size
        print('Removing {} items from class {}'. format(class_items_to_remove[class_label], class_label))

    for i, item in enumerate(zip(undersampled_x, undersampled_y)):
        x, y = item
        if class_items_to_remove[y] > 0:
            del undersampled_x[i]
            del undersampled_y[i]
            class_items_to_remove[y] -= 1

    return undersampled_x, undersampled_y


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

    # Shuffle dataset
    dataset = dataset.sample(frac=1)

    # # Limit number of examples for given classes
    # g = dataset.groupby('sentiment')
    # dataset = g.apply(lambda x: x.sample(g.size().max()).reset_index(drop=True))
    #
    # # Shuffle dataset again
    # dataset = dataset.sample(frac=1)

    tweets = dataset['content']
    sentiment = dataset['sentiment']

    # Print count of tweets from each class
    for emotion in EMOTION_LABELS_MAP.keys():
        emotion_tweets = [t[1] for t, s in zip(tweets.iteritems(), sentiment.iteritems()) if s[1] == emotion]
        print('Count of tweets with class {}: {}'.format(emotion, len(emotion_tweets)))

    # Preprocessing

    # Handling exclamation marks
    exclamation_mark_regex = re.compile(r'!+')
    tweets = [exclamation_mark_regex.sub(' !', tweet) for tweet in tweets]

    tokenizer = Tokenizer(filters='"#$%&*+,-.<=>?@[\\]^_`{}~\t\n')
    tokenizer.fit_on_texts(tweets)

    # Dump fitted tokenizer
    with open(args.tokenizer_path, 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    preprocessed_texts = tokenizer.texts_to_sequences(tweets)

    # Reduce labels, remove tweets from unwanted categories
    preprocessed_texts, text_labels = zip(*((preproc_text, EMOTION_LABELS_MAP[sentiment_label])
                                            for preproc_text, sentiment_label in
                                            zip(preprocessed_texts, sentiment)
                                            if EMOTION_LABELS_MAP[sentiment_label] is not None))

    text_labels = to_categorical(text_labels)

    # Padding
    preprocessed_texts = sequence.pad_sequences(preprocessed_texts, maxlen=MAX_WORDS)

    embedding_index = create_glove_embedding_index(args.glove_embeddings_path)
    embedding_matrix = create_embedding_matrix(word_index=tokenizer.word_index,
                                               embeddings_index=embedding_index,
                                               glove_embeddings_dim=args.glove_embeddings_dim)

    # SMOTE
    USE_SMOTE = False
    if USE_SMOTE:
        # Construct text representation vectors

        trv_start = time.time()
        trvs = create_text_representation_vectors(texts=preprocessed_texts,
                                                  embedding_matrix=embedding_matrix)
        print('TRVS created! Time elapsed: {} seconds'.format(time.time() - trv_start))

        # Split train/test data
        trvs = numpy.array(trvs)

        print(numpy.shape(trvs))
        sm = smote.SMOTE()
        # sm = adasyn.ADASYN(n_jobs=4, ratio='minority')


        # show_pca_and_lda_plots(numpy.array(trvs), numpy.array(smote_labels))
        # show_tsne_plot(numpy.array(trvs), numpy.array(smote_labels))

        # print(numpy.shape(numpy.array(trvs)))
        # show_pca_and_lda_plots(numpy.array(trvs), smote_labels)

        # with open('data_before_smote.csv', mode='w', encoding='utf-8') as data_before_smote_file:
        #     data = ''.join('{} : {} \n'.format(label, ','.join(str(vec) for vec in trv))
        #                    for label, trv in zip(smote_labels, trvs))
        #     data_before_smote_file.write(data)
        #
        # print('Original data saved')
        x_train, x_test, y_train, y_test = train_test_split(trvs, text_labels,
                                                            test_size=0.20,
                                                            random_state=SEED)

        smote_start = time.time()
        smote_labels = [numpy.argmax(label) for label in y_train]
        x_train, y_train = sm.fit_sample(x_train, smote_labels)
        print('SMOTE done! Time elapsed: {} seconds'.format(time.time() - smote_start))

        print(numpy.shape(x_train))

        # show_pca_and_lda_plots(x_res, y_res)

        # Expand x_res to 3 dimensions
        x_train = numpy.expand_dims(x_train, axis=2)
        x_test = numpy.expand_dims(x_test, axis=2)
        # Convert text_labels back to categorical
        y_train = to_categorical(y_train)

        model = glove_model_trv(trv_size=args.glove_embeddings_dim)

    else:
        preprocessed_texts = numpy.array(preprocessed_texts)
        text_labels = numpy.array(text_labels)
        x_train, x_test, y_train, y_test = train_test_split(preprocessed_texts, text_labels,
                                                            test_size=0.20,
                                                            random_state=SEED)

        # # Show test data
        # index_word = {v: k for k, v in tokenizer.word_index.items()}  # map back
        # for tokenized_text, label in zip(x_test, y_test):
        #     if numpy.argmax(label) in {4, 5}:
        #         print(numpy.argmax(label))
        #         text = ' '.join(index_word[token] for token in tokenized_text if token != 0)
        #         print(text)

        # exit(0)

        duplicate_oversampling_start = time.time()
        singular_y_train_labels = [numpy.argmax(label) for label in y_train]
        x_train, y_train = random_undersampling(x_train, singular_y_train_labels, ratio=0.7)
        y_train = to_categorical(y_train)
        x_train, y_train = numpy.array(x_train), numpy.array(y_train)
        print('Duplicate oversampling done! Time elapsed: {} seconds'.format(time.time()
                                                                             - duplicate_oversampling_start))

        model = glove_model_multikernel(input_dim=len(tokenizer.word_index) + 1,
                                        embedding_matrix=embedding_matrix,
                                        embedding_dim=args.glove_embeddings_dim,
                                        input_length=MAX_WORDS)

    print('Build model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[categorical_accuracy,
                           top_2_categorical_accuracy])
    print(model.summary())

    print(x_train[0])
    print(y_train[0])

    PLOT_MODEL = True
    if PLOT_MODEL:
        import os
        os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        plot_model(model, to_file='model_{}.png'.format(time.time()))

    model.fit(x=x_train, y=y_train,
              validation_data=(x_test, y_test),
              epochs=args.epochs,
              batch_size=256,
              verbose=1,
              callbacks=[CSVLogger('training_{time}.csv'.format(time=time.time())),
                         EarlyStopping(monitor='val_loss', min_delta=0.001,
                                       patience=8, verbose=1, mode='auto')])

    # Evaluate model (on test data)
    predictions = model.predict(x_test, verbose=0)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true=[numpy.argmax(v) for v in y_test],
                                   y_pred=[numpy.argmax(pred) for pred in predictions])
    print(conf_matrix)

    # classification metrics
    report = classification_report(y_true=[numpy.argmax(v) for v in y_test],
                                   y_pred=[numpy.argmax(pred) for pred in predictions])
    print(report)

    model.save(args.model_path)
