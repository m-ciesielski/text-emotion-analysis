import os

import inference.inference_cnn

for tsa_dataset in os.listdir('tsa_datasets'):
    os.environ['DATASET_PATH'] = 'tsa_datasets/{}'.format(tsa_dataset)
    os.environ['OUTPUT_PATH'] = 'tsa_datasets_emotion_v2/{}'.format(tsa_dataset)
    print('Running emotion analysis on {}'.format(tsa_dataset))
    inference.inference_cnn.main()
    print('Finished emotion analysis on {}'.format(tsa_dataset))
