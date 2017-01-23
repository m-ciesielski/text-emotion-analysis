import pickle

import numpy.random
import pandas
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

# fix random seed for reproducibility
SEED = 7
DATASET_PATH = 'text_emotion.csv'
ngram_range = 1
max_features = 50000


def get_emotion_from_categorical(categorical):
    for i in range(len(categorical)):
        if categorical[i] == 1:
            return i

numpy.random.seed(SEED)

# load the dataset
dataset = pandas.read_csv(DATASET_PATH)

tweets = dataset['content']
emotions = dataset['sentiment']

# Preprocessing
# Tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)

# Dump fitted tokenizer
with open('tokenizer.bin', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

preprocessed_texts = tokenizer.texts_to_sequences(tweets)

# Emotions
emotions_tokens = {'love': 0, 'enthusiasm': 1, 'happiness': 2, 'fun': 3, 'relief': 4, 'surprise': 5,
                   'neutral': 6, 'empty': 7, 'boredom': 8, 'worry': 9, 'sadness': 10, 'anger': 11,
                   'hate': 12}
emotions = [emotions_tokens[e] for e in emotions]
emotions = to_categorical(emotions)

# Padding
max_words = 37
preprocessed_texts = sequence.pad_sequences(preprocessed_texts, maxlen=max_words)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32, input_length=max_words))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(13, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(preprocessed_texts, emotions, validation_split=0.3, nb_epoch=18, batch_size=128, verbose=1)

# Final evaluation of the model
scores = model.evaluate(preprocessed_texts, emotions, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


## Write predictions results to a file
#predictions = model.predict_classes(preprocessed_texts)
#with open('predictions.txt', 'w', encoding='utf-8') as predictions_file:
#    for i, pred in enumerate(predictions):
#        predictions_file.write('Correct: {correct}, Predicted {pred},'
#                               ' Actual {actual}: {text}\n'.format(
#            correct=bool(pred == get_emotion_from_categorical(emotions[i])),
#            pred=pred,
#            actual=get_emotion_from_categorical(emotions[i]),
#            text=tweets.iloc[i]))
#
## Save model
model.save('emotion_analysis_CNN.h5')
