import os

import inference.sentiment_analysis_v2

for tsa_dataset in os.listdir('tsa_datasets'):
    os.environ['DATASET_PATH'] = 'tsa_datasets/{}'.format(tsa_dataset)
    os.environ['OUTPUT_PATH'] = 'tsa_datasets_sentiment_v2/{}'.format(tsa_dataset)
    print('Running seniment analysis on {}'.format(tsa_dataset))
    inference.sentiment_analysis_v2.main()
    print('Finished seniment analysis on {}'.format(tsa_dataset))
