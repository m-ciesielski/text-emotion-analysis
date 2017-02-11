import pickle

import numpy
from keras.models import load_model
from keras.preprocessing import sequence

SEED = 7
numpy.random.seed(SEED)

emotion_labels = {0: 'love', 1: 'enthusiasm', 2: 'happiness', 3: 'fun', 4: 'relief', 5: 'surprise',
                  6: 'neutral', 7:  'empty', 8: 'boredom', 9: 'worry', 10: 'sadness', 11: 'anger',
                  12: 'hate'}


def get_prediction_accuracy(prediction_array):
    return numpy.max(prediction_array)


def get_predicted_emotion(prediction_array):
    predicted_class_index = numpy.argmax(prediction_array)
    return emotion_labels[predicted_class_index]

ngram_range = 1
max_features = 50000

# load model
model = load_model('emotion_analysis_CNN_new.h5')

# Prepare test data
test_tweets = ['I hate Trump',
               'Trump is becoming president tomorrow. We voted for him as a country and we need to accept it',
               'Trump is our president now and you have to respect him!',
               'dammit! hulu desktop has totally screwed up my ability to talk to a particular port on one of our dev servers. so i can\'t watch and code',
               'blablablalbla',
               'It is so annoying when she starts typing on her computer in the middle of the night!',
               'I hate Bakersfield and I hate the Ports, let me go home already.  I want to start my vacation.']

# Load dumped tokenizer
with open('tokenizer.bin', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

preprocessed_test_tweets = tokenizer.texts_to_sequences(test_tweets)

# Padding
max_words = 37
preprocessed_test_tweets = sequence.pad_sequences(preprocessed_test_tweets, maxlen=max_words)

predictions = model.predict(numpy.array(preprocessed_test_tweets))

for i, p in enumerate(predictions):
    print('Emotion: {}, Accuracy: {:.2f}, Text: {}'.format(get_predicted_emotion(p), get_prediction_accuracy(p),
                                                          test_tweets[i]))
