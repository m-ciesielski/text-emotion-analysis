import csv


class PreprocessedTweet:
    def __init__(self, text, timestamp, hashtags=None, retweet_count=None):
        self.text = text
        self.timestamp = timestamp
        if hashtags:
            self.hashtags = list(map(lambda h: h.get('text'), hashtags))
        self.retweet_count = retweet_count

    def save_in_csv(self, csv_path):
        with open(csv_path, "a", encoding='utf-8') as csv_file:
            tweet_writer = csv.writer(csv_file, delimiter=' ',
                                      quotechar='|', quoting=csv.QUOTE_MINIMAL)
            tweet_writer.writerow([self.text, self.hashtags, self.retweet_count, self.timestamp])