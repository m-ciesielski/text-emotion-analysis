import glob
import datetime

import matplotlib.pyplot as pyplot
import matplotlib.dates as mdates
import yaml


class EmotionAnalysisResult:
    def __init__(self, result_yaml):
        self.results_percentage = {'love': 0, 'enthusiasm': 0, 'happiness': 0, 'fun': 0, 'relief': 0, 'surprise': 0,
                                   'neutral': 0, 'empty': 0, 'boredom': 0, 'worry': 0, 'sadness': 0, 'anger': 0,
                                   'hate': 0}
        for result in result_yaml['Weighted results percentage']:
            self.results_percentage[result['Emotion']] = result['weighted result percentage']

        self.date = datetime.datetime.strptime(result_yaml['date'], '%d/%m/%Y').date()

RESULTS_PATH = 'emotion_results/*.txt'

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
    results.append(EmotionAnalysisResult(result_yaml))

results.sort(key=lambda x: x.date)

percentage_results = {'love': 0, 'enthusiasm': 0, 'happiness': 0, 'fun': 0, 'relief': 0, 'surprise': 0,
                                   'neutral': 0, 'empty': 0, 'boredom': 0, 'worry': 0, 'sadness': 0, 'anger': 0,
                                   'hate': 0}

for key in percentage_results:
    percentage_results[key] = [result.results_percentage[key] for result in results]

pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
pyplot.gca().xaxis.set_major_locator(mdates.DayLocator())

dates = [result.date for result in results]

for key in percentage_results:
    pyplot.plot(dates, percentage_results[key], label=key)

pyplot.legend()

pyplot.show()





