import argparse
import os
import pickle

import numpy
import pandas
from keras.models import load_model
from keras.preprocessing import sequence

from training.train_cnn import top_2_categorical_accuracy,\
    EMOTION_VALUES_MAP, EMOTION_LABELS_MAP,\
    create_text_representation_vectors, create_glove_embedding_index

SEED = 7
numpy.random.seed(SEED)


def get_predicted_emotion(prediction_array):
    predicted_class_index = numpy.argmax(prediction_array)
    return EMOTION_VALUES_MAP[predicted_class_index]


def get_top_2_predicted_emotions(prediction_array):
    top_2_indices = numpy.argpartition(prediction_array, -2)[-2:]
    return EMOTION_VALUES_MAP[top_2_indices[0]], EMOTION_VALUES_MAP[top_2_indices[1]]


def get_prediction_accuracy(prediction_array):
    return numpy.max(prediction_array)


def preprocess_tweets(tweets: list, tokenizer, max_words=37) -> numpy.ndarray:
    # tweets = [t for t in tweets if 'https://t.co' not in t]
    preprocessed_tweets = tokenizer.texts_to_sequences(tweets)
    preprocessed_tweets = sequence.pad_sequences(preprocessed_tweets, maxlen=max_words)
    # embedding_index = create_glove_embedding_index(glove_embeddings_file_path=glove_embeddings_path)
    # trvs = create_text_representation_vectors(texts=preprocessed_tweets,
    #                                           word_index=tokenizer.word_index,
    #                                           embeddings_index=embedding_index,
    #                                           glove_embeddings_dim=glove_embeddings_dim)
    # Change trvs to three dimensional sequence of vectors
    preprocessed_tweets = numpy.array(preprocessed_tweets)
    return preprocessed_tweets


def load_tokenizer(tokenizer_path: str):
    with open(tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    return tokenizer


def analyze_tweets(model, dataset_path: str, tokenizer_path: str):
    dataset = pandas.read_csv(dataset_path, delimiter=' ', quotechar='|')
    tweets = dataset['Text']

    # Preprocess texts
    with open(tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    trvs = preprocess_tweets(tweets=tweets, tokenizer=tokenizer)

    # Analyze
    predicted_classes = model.predict_classes(trvs, verbose=0)
    prediction_confidence = []
    for scores in model.predict(trvs, verbose=0):
        prediction_confidence.append(numpy.max(scores))
    results = zip(predicted_classes, prediction_confidence, tweets)

    # Calculate weighted results
    weighted_results = EMOTION_LABELS_MAP

    for result in results:
        weighted_results[EMOTION_VALUES_MAP[result[0]]] += result[1]

    weighted_results_sum = 0
    print('Weighted results:')
    for key in weighted_results:
        print('  - Emotion: {0}\n    weighted result: {1}'.format(key, weighted_results[key]))
        weighted_results_sum += weighted_results[key]

    print('Weighted results percentage:')
    for key in weighted_results:
        print('  - Emotion: {0}\n    weighted result percentage: {1}'.format(key,
                                                                     (weighted_results[key]/weighted_results_sum)*100))


def write_predictions_to_file(file_path, model, preprocessed_tweets, dataset):
    predictions = model.predict(preprocessed_tweets, verbose=1)

    primary_emotions = []
    secondary_emotions = []
    prediction_accuracy = []

    for i, pred in enumerate(predictions):
        primary_emotion, secondary_emotion = get_top_2_predicted_emotions(pred)
        primary_emotions.append(primary_emotion)
        secondary_emotions.append(secondary_emotion)
        prediction_accuracy.append(get_prediction_accuracy(pred))

    dataset['primary_emotion'] = primary_emotions
    dataset['secondary_emotion'] = secondary_emotions
    dataset['prediction_accuracy'] = prediction_accuracy
    dataset.to_csv(file_path, encoding='utf-8', sep=' ', quotechar='|', index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Perform emotion analysis on a given dataset.')
    parser.add_argument('-d', '--dataset-path', default=os.environ.get('DATASET_PATH'),
                        type=str, help='Path to dataset CSV file.')
    parser.add_argument('-m', '--model-path', default='ea_cnn_we.h5',
                        type=str, help='Path to a file with trained emotion analysis model.')
    parser.add_argument('-t', '--tokenizer-path', default='tokenizer.pkl',
                        type=str, help='Path to a serialized keras.preprocessing.text.Tokenizer object'
                                       'used during model training.')
    parser.add_argument('-o', '--output-path', default=os.environ.get('OUTPUT_PATH'),
                        type=str, help='Path to a serialized keras.preprocessing.text.Tokenizer object'
                                       'used during model training.')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = pandas.read_csv(args.dataset_path, sep=' ', quotechar='|', encoding='utf-8')
    texts = dataset['Text']
    model = load_model(args.model_path, custom_objects={'top_2_categorical_accuracy': top_2_categorical_accuracy})
    tokenizer = load_tokenizer(args.tokenizer_path)
    preprocessed_tweets = preprocess_tweets(texts, tokenizer)
    # analyze_tweets(preprocessed_tweets, args.tokenizer_path)
    write_predictions_to_file(args.output_path, model, preprocessed_tweets, dataset)

if __name__ == '__main__':
    main()
