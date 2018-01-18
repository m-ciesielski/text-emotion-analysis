# twitter-sentiment-analysis

This repository contains scripts for:
* Collecting tweets from Twitter Streaming API
* Performing sentiment analysis on collected data
* Training Convolutional Neural Network for emotion classification
* Using trained CNN model for emotion classification

## Installation

### Prerequisites
`Python >= 3.5`
`virtualenv`

In order to use script for tweets collection, copy `collect/config_example.ini`
file to `collect/config.ini` and fill it with your Twitter application credentials.


### Linux
Install required dependencies:
`pip install -r requirements.txt`

### Windows
Anaconda or Miniconda is recommended for running this project on Windows.

Create new Conda environment and install Conda dependencies:
`conda install --file conda_requirements.txt`

And install pip dependecies with:
`pip install -r requirements.txt`
