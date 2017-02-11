import argparse
import pickle
import csv

import numpy
import pandas
from keras.models import load_model
from keras.preprocessing import sequence

SEED = 7
numpy.random.seed(SEED)

emotion_labels = {0: 'love', 1: 'enthusiasm', 2: 'happiness', 3: 'fun', 4: 'relief', 5: 'surprise',
                  6: 'neutral', 7:  'empty', 8: 'boredom', 9: 'worry', 10: 'sadness', 11: 'anger',
                  12: 'hate'}


def get_tweets_from_dataset(dataset_path: str):
    dataset = pandas.read_csv(dataset_path, delimiter=' ', quotechar='|')
    return dataset['Text']


def load_tokenizer(tokenizer_path: str):
    with open(tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    return tokenizer


def get_emotion_from_categorical(categorical):
    for i in range(len(categorical)):
        if categorical[i] == 1:
            return i


def get_predicted_emotion(prediction_array):
    predicted_class_index = numpy.argmax(prediction_array)
    return emotion_labels[predicted_class_index]


def get_prediction_accuracy(prediction_array):
    return numpy.max(prediction_array)


def preprocess_tweets(tweets: list, tokenizer, max_words=37) -> list:
    preprocessed_tweets = tokenizer.texts_to_sequences(tweets)
    preprocessed_tweets = sequence.pad_sequences(preprocessed_tweets, maxlen=max_words)
    return preprocessed_tweets


def analyze_tweets(model, preprocessed_tweets: list):
    # Analyze
    predicted_classes = model.predict_classes(numpy.array(preprocessed_tweets), verbose=0)
    prediction_confidence = []
    for scores in model.predict(preprocessed_tweets, verbose=0):
        prediction_confidence.append(numpy.max(scores))
    results = zip(predicted_classes, prediction_confidence, preprocessed_tweets)

    # Calculate weighted results
    weighted_results = {'love': 0, 'enthusiasm': 0, 'happiness': 0, 'fun': 0, 'relief': 0, 'surprise': 0,
                        'neutral': 0, 'empty': 0, 'boredom': 0, 'worry': 0, 'sadness': 0, 'anger': 0,
                        'hate': 0}

    for result in results:
        weighted_results[emotion_labels[result[0]]] += result[1]

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
            result_writer.writerow([get_predicted_emotion(pred), '{0:.2f}'.format(get_prediction_accuracy(pred)),
                                    tweets.iloc[i]])


def parse_args():
    parser = argparse.ArgumentParser(description='Perform emotion analysis on a given dataset of tweets.')
    parser.add_argument('-d', '--dataset-path', required=True,
                        type=str, help='Path to dataset CSV file.')
    parser.add_argument('-m', '--model-path', default='emotion_analysis_CNN.h5',
                        type=str, help='Path to a file with trained emotion analysis model.')
    parser.add_argument('-t', '--tokenizer-path', default='tokenizer.bin',
                        type=str, help='Path to a serialized keras.preprocessing.text.Tokenizer object'
                                       'used during model training.')
    return parser.parse_args()


def main():
    args = parse_args()
    tweets = get_tweets_from_dataset(args.dataset_path)
    tokenizer = load_tokenizer(args.tokenizer_path)
    model = load_model(args.model_path)

    preprocessed_tweets = preprocess_tweets(tweets, tokenizer)
    analyze_tweets(model, preprocessed_tweets)
    write_predictions_to_file('ea_results.txt', model, preprocessed_tweets, tweets)

if __name__ == '__main__':
    main()
