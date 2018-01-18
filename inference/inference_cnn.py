import argparse
import csv
import pickle

import numpy
import pandas
from keras.models import load_model
from keras.preprocessing import sequence

from training.train_cnn import top_2_categorical_accuracy, emotion_indices, EMOTION_LABELS_MAP

SEED = 7
numpy.random.seed(SEED)


def get_tweets_from_dataset(dataset_path: str):
    dataset = pandas.read_csv(dataset_path, delimiter=' ', quotechar='|')
    return dataset['Text']


def get_predicted_emotion(prediction_array):
    predicted_class_index = numpy.argmax(prediction_array)
    return emotion_indices[predicted_class_index]


def get_top_2_predicted_emotions(prediction_array):
    top_2_indices = numpy.argpartition(prediction_array, -2)[-2:]
    return emotion_indices[top_2_indices[0]], emotion_indices[top_2_indices[1]]


def get_prediction_accuracy(prediction_array):
    return numpy.max(prediction_array)


def preprocess_tweets(tweets: list, tokenizer, max_words=37) -> list:
    # tweets = [t for t in tweets if 'https://t.co' not in t]
    preprocessed_tweets = tokenizer.texts_to_sequences(tweets)
    preprocessed_tweets = sequence.pad_sequences(preprocessed_tweets, maxlen=max_words)
    return preprocessed_tweets


def load_tokenizer(tokenizer_path: str):
    with open(tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    return tokenizer


def analyze_tweets(model, dataset_path: str, tokenizer_path: str, max_words=37):
    dataset = pandas.read_csv(dataset_path, delimiter=' ', quotechar='|')
    tweets = dataset['Text']

    # Preprocess texts
    with open(tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    preprocessed_tweets = tokenizer.texts_to_sequences(tweets)

    # Padding
    preprocessed_tweets = sequence.pad_sequences(preprocessed_tweets, maxlen=max_words)

    # Analyze
    predicted_classes = model.predict_classes(numpy.array(preprocessed_tweets), verbose=0)
    prediction_confidence = []
    for scores in model.predict(numpy.array(preprocessed_tweets), verbose=0):
        prediction_confidence.append(numpy.max(scores))
    results = zip(predicted_classes, prediction_confidence, tweets)

    # Calculate weighted results
    weighted_results = EMOTION_LABELS_MAP

    for result in results:
        weighted_results[emotion_indices[result[0]]] += result[1]

    weighted_results_sum = 0
    print('Weighted results:')
    for key in weighted_results:
        print('  - Emotion: {0}\n    weighted result: {1}'.format(key, weighted_results[key]))
        weighted_results_sum += weighted_results[key]

    print('Weighted results percentage:')
    for key in weighted_results:
        print('  - Emotion: {0}\n    weighted result percentage: {1}'.format(key,
                                                                     (weighted_results[key]/weighted_results_sum)*100))


def write_predictions_to_file(file_path, model, preprocessed_tweets, tweets):
    predictions = model.predict(numpy.array(preprocessed_tweets))
    assert len(predictions) == len(preprocessed_tweets)
    with open(file_path, "w", encoding='utf-8') as csv_file:
        result_writer = csv.writer(csv_file, delimiter=' ',
                                   quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(['Emotion', 'Accuracy', 'Text'])
        for i, pred in enumerate(predictions):
            result_writer.writerow([get_top_2_predicted_emotions(pred),
                                    '{0:.2f}'.format(get_prediction_accuracy(pred)),
                                    tweets.iloc[i]])


def parse_args():
    parser = argparse.ArgumentParser(description='Perform emotion analysis on a given dataset.')
    parser.add_argument('-d', '--dataset-path', required=True,
                        type=str, help='Path to dataset CSV file.')
    parser.add_argument('-m', '--model-path', default='emotion_analysis_CNN_keras_2.h5',
                        type=str, help='Path to a file with trained emotion analysis model.')
    parser.add_argument('-t', '--tokenizer-path', default='tokenizer.bin',
                        type=str, help='Path to a serialized keras.preprocessing.text.Tokenizer object'
                                       'used during model training.')
    parser.add_argument('-o', '--output-path', default='inference_results.csv',
                        type=str, help='Path to a serialized keras.preprocessing.text.Tokenizer object'
                                       'used during model training.')
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model_path, custom_objects={'top_2_categorical_accuracy': top_2_categorical_accuracy})
    tokenizer = load_tokenizer(args.tokenizer_path)
    tweets = get_tweets_from_dataset(args.dataset_path)
    preprocessed_tweets = preprocess_tweets(tweets, tokenizer)
    # analyze_tweets(model, args.dataset_path, args.tokenizer_path)
    write_predictions_to_file(args.output_path, model, preprocessed_tweets, tweets)

if __name__ == '__main__':
    main()
