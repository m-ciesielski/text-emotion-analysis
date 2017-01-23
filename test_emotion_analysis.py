import pickle

import numpy
from keras.models import load_model
from keras.preprocessing import sequence

SEED = 7
numpy.random.seed(SEED)

ngram_range = 1
max_features = 50000

# load model
model = load_model('emotion_analysis_CNN.h5')

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

test_tweets = tokenizer.texts_to_sequences(test_tweets)

# Padding
max_words = 37
test_tweets = sequence.pad_sequences(test_tweets, maxlen=max_words)

predictions = model.predict(numpy.array(test_tweets))

print(model.predict_classes(numpy.array(test_tweets)))
