import argparse
import pickle
import time
import os

from imblearn.over_sampling import smote, adasyn
import numpy
import pandas
import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.metrics import top_k_categorical_accuracy
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

    trv_start = time.time()
    trvs = create_text_representation_vectors(texts=preprocessed_texts,
                                              word_index=tokenizer.word_index,
                                              embeddings_index=embedding_index,
                                              glove_embeddings_dim=args.glove_embeddings_dim)
    print('TRVS created! Time elapsed: {} seconds'.format(time.time() - trv_start))

    # Split train/test data
    trvs = numpy.array(trvs)

    print(numpy.shape(trvs))

    # SMOTE
    sm = smote.SMOTE()
    # sm = adasyn.ADASYN(n_jobs=4, ratio='minority')
    smote_labels = [numpy.argmax(label) for label in text_labels]

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

    smote_start = time.time()
    x_res, y_res = sm.fit_sample(trvs, smote_labels)
    print('SMOTE done! Time elapsed: {} seconds'.format(time.time() - smote_start))

    print(numpy.shape(x_res))

    # Save data before and after performing SMOTE for further analysis
    #
    # with open('data_after_smote.csv', mode='w') as data_after_smote_file:
    #     for label, trv in zip(y_res, x_res):
    #         data_after_smote_file.write('{} : {} \n'.format(label, ','.join(str(vec) for vec in trv)))
    #
    #     data_after_smote_file.flush()
    #
    # exit(0)

    #show_tsne_plot(x_res, y_res)

    print(numpy.shape(x_res))

    # show_pca_and_lda_plots(x_res, y_res)

    # Expand x_res to 3 dimensions
    #x_res = numpy.expand_dims(x_res, axis=2)
    # Convert text_labels back to categorical
    # y_res = to_categorical(y_res)

    x_train, x_test, y_train, y_test = train_test_split(x_res, y_res,
                                                        test_size=0.20,
                                                        random_state=SEED)

    print(x_train[0])
    print(y_train[0])

    model = GaussianNB()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    conf_matrix = confusion_matrix(y_true=[y for y in y_test],
                                   y_pred=[pred for pred in predictions])
    print(conf_matrix)
    print('Acc: {}'.format(sum(1 if pred == y else 0
                               for y, pred in zip(y_test, predictions))/len(y_test) * 100.0))
