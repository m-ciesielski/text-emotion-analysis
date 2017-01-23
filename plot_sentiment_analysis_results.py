import glob
import datetime

import matplotlib.pyplot as pyplot
import matplotlib.dates as mdates
import yaml


class SentimentAnalysisResult:
    def __init__(self, result_yaml):
        self.negative_sentiment_percentage = result_yaml['Negative tweets ratio'].replace('%', '')
        self.positive_sentiment_percentage = result_yaml['Positive tweets ratio'].replace('%', '')
        self.negative_sentiment_score_ratio = result_yaml['Negative sentiment score percentage']
        self.positive_sentiment_score_ratio = result_yaml['Positive sentiment score percentage']
        self.date = datetime.datetime.strptime(result_yaml['date'], '%d/%m/%Y').date()

RESULTS_PATH = 'results/*.txt'

files = glob.glob(RESULTS_PATH)

result_yamls = []
for file in files:
    try:
        with open(file, 'r') as f:
            result_yamls.append(yaml.load(f))
    except IOError:
        print('Failed to read {0} file.'.format(file))

print(result_yamls)

results = []
for result_yaml in result_yamls:
    results.append(SentimentAnalysisResult(result_yaml))

results.sort(key=lambda x: x.date)


negative_sentiment_percentage = [result.negative_sentiment_percentage for result in results]
positive_sentiment_percentage = [result.positive_sentiment_percentage for result in results]
negative_sentiment_score_ratio = [result.negative_sentiment_score_ratio for result in results]
positive_sentiment_score_ratio = [result.positive_sentiment_score_ratio for result in results]
dates = [result.date for result in results]

pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
pyplot.gca().xaxis.set_major_locator(mdates.DayLocator())

pyplot.plot(dates, negative_sentiment_percentage, label='Procent negatywnych tweetów', color='red')
pyplot.plot(dates, positive_sentiment_percentage, label='Procent pozytywnych tweetów', color='green')
pyplot.plot(dates, negative_sentiment_score_ratio, label='Procent negatywnej polaryzacji', color='pink')
pyplot.plot(dates, positive_sentiment_score_ratio, label='Procent pozytywnej polaryzacji', color='blue')

pyplot.legend()

pyplot.show()





