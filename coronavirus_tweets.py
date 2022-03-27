import copy
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import sys
import re
import sklearn.naive_bayes as nb
import requests

DATA_FILE = 'coronavirus_tweets.csv'
OUTPUT_FILE = 'log.txt'


# Part 3: Mining text data.

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_3(data_file):
	try:
		# open data file in csv format
		df = pd.read_csv(data_file, na_filter=False, encoding='latin-1')
	# handle exceptions:
	except IOError as ioe:
		print('there was an I/O error trying to open the data file: ' + str(ioe))
		sys.exit()
	except Exception as ex:
		print('there was an error: ' + str(ex))
		sys.exit()
	# drop attributes Channel and Region
	return df


# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	return df['Sentiment'].value_counts().index.tolist()


# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	return df['Sentiment'].value_counts().index[1]


# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	exp_df = df.loc[df['Sentiment'] == 'Extremely Positive']
	return exp_df['TweetAt'].value_counts().index[0]


# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.lower()


# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))


# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: ' '.join(x.split()))


# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: x.split())


# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	num_words = 0
	for tweets in tdf['OriginalTweet']:
		num_words += len(tweets)
	return num_words


# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	num_words = 0
	words = []
	for tweets in tdf['OriginalTweet']:
		words += tweets
	num_words += len(set(words))
	return num_words


# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	words = []
	for tweets in tdf['OriginalTweet']:
		words += tweets
	num_words = pd.Series(words).value_counts()
	freq_words = num_words.index[0:k].tolist()
	return freq_words


# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	# download stop words list via url
	url = 'https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt'
	file = requests.get(url)
	file_name = './english-stop-words-large.txt'
	open(file_name, 'wb').write(file.content)
	# load english-stop-words-large.txt into list
	f1 = open(file_name, 'r')
	txt = f1.readlines()
	stop_words = []
	for line in txt:
		stop_words.append(line.replace('\n', ''))

	# remove stop words
	new_tweets = []
	for tweet in tdf['OriginalTweet']:
		tweet = list(filter(lambda x: x not in stop_words, tweet))
		tweet = list(filter(lambda x: len(x) > 2, tweet))
		new_tweets.append(tweet)
	tdf['OriginalTweet'] = pd.Series(new_tweets)


# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	stemmer = SnowballStemmer('english')
	stem_list = []
	for tweet in tdf['OriginalTweet']:
		stemmed_words = [stemmer.stem(word) for word in tweet]
		stem_list.append(stemmed_words)
	tdf['OriginalTweet'] = pd.Series(stem_list)


# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	copy_df = copy.copy(df)
	# preprocess train data
	lower_case(copy_df)
	remove_non_alphabetic_chars(copy_df)
	remove_multiple_consecutive_whitespaces(copy_df)

	x_train = copy_df['OriginalTweet'].values.tolist()
	y_train = np.array(copy_df['Sentiment'].values.tolist())
	# extract features
	vec = CountVectorizer(ngram_range=(3, 4), stop_words='english')
	x_train = vec.fit_transform(x_train)

	# train classifier
	clf = nb.MultinomialNB()
	clf.fit(x_train, y_train)
	return clf.predict(x_train)


# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	m = len(y_pred)
	correct_num = sum(y_pred == y_true)
	accuracy = correct_num/m
	return round(accuracy, 3)


# MAIN
# print log in the file
f = open(OUTPUT_FILE, 'a', encoding='utf-8')
sys.stdout = f
sys.stderr = f
print("------------------Part 3--------------------")
print('-------------orginal data----------------')
df = read_csv_3(DATA_FILE)
odf = copy.copy(df)
print(df)
print('-------------possible sentiment----------------')
print(get_sentiments(df))
print('-------------second most popular----------------')
print(second_most_popular_sentiment(df))
print('-------------most popular date----------------')
print(date_most_popular_tweets(df))
print('-------------lower case tweet----------------')
lower_case(df)
print(df['OriginalTweet'])
print('-------------remove non-character----------------')
remove_non_alphabetic_chars(df)
print(df['OriginalTweet'])
print('-------------remove multiple whitespace----------------')
remove_multiple_consecutive_whitespaces(df)
print(df['OriginalTweet'])
print('-------------tokenize----------------')
tokenize(df)
print(df['OriginalTweet'])
print('-------------word counts repetition----------------')
print(count_words_with_repetitions(df))
print('--------word counts without repetition----------')
print(count_words_without_repetitions(df))
print('--------frequent words----------')
print(frequent_words(df, 10))
print('--------remove stop words----------')
ndf = copy.copy(df)
remove_stop_words(ndf)
print(count_words_with_repetitions(ndf))
print(count_words_without_repetitions(ndf))
print(frequent_words(ndf, 10))
print(ndf['OriginalTweet'])
print('-------------stemming-------------')
stemming(ndf)
print(ndf['OriginalTweet'])
print('-------------predict-------------')
predict = mnb_predict(odf)
print(predict)
print('-------------accuracy-------------')
true_label = np.array(df['Sentiment'])
print(mnb_accuracy(predict, true_label))


