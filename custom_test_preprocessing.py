import argparse
import pandas as pd 
import numpy as np
import time as t
import re
from nltk.stem import PorterStemmer
from sklearn.utils import resample, shuffle
pd.set_option('mode.chained_assignment', None)

import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer

import spacy
nlp = spacy.load("en_core_web_sm")

ps = PorterStemmer()

def create_arg_parser():
	parser = argparse.ArgumentParser()
	# -cr flag to run the algorithm with custom features
	parser.add_argument("-cr", "--create_custom_features", action="store_true",
						help="Create custom feature matrix and train the svm model")
	
	args = parser.parse_args()
	return args

#from read_json import test_df

test_df = pd.read_csv('./data/custom_test.csv')


news_dict = {
	'The Australian': 'Australia',
	'Sydney Morning Herald (Australia)': 'Herald',
	'The Age (Melbourne, Australia)': 'Age',
	'The Times of India (TOI)': 'India',
	'The Hindu': 'Hindu',
	'The Times (South Africa)': 'Times',
	'Mail & Guardian': 'Guardian',
	'The Washington Post': 'Washington',
	'The New York Times': 'Newyork'
	}

political_orientation = {
	'The Australian': 'RC',
	'Sydney Morning Herald (Australia)': 'LC',
	'The Age (Melbourne, Australia)': 'LC',
	'The Times of India (TOI)': 'RC',
	'The Hindu': 'LC',
	'The Times (South Africa)': 'RC',
	'Mail & Guardian': 'LC',
	'The Washington Post': 'LC',
	'The New York Times': 'LC'
	}

def get_orientation(x):
	return political_orientation.get(x)

def get_key(x):
	return news_dict.get(x)

test_df['news_key'] = np.vectorize(get_key)(test_df['newspaper_name'])

test_df['political_orientation'] = np.vectorize(get_orientation)(test_df['newspaper_name'])


def apply_stem(body):
	words = [ps.stem(w) for w in body.split()]
	return ' '.join(words)


def remove_url(body):
	_body = re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?','',body)	 
	return _body

def remove_newline(body):
	_body = re.sub('\n', '', body)
	return _body
	
def apply_lemma(body):
	doc = nlp(body)
	lemma = [token.lemma_ for token in doc]
	return ' '.join(lemma)
	

def remove_stopwords(body):
	doc = nlp(body)
	stop = [token.text for token in doc if not token.is_stop and not token.is_punct]
	return ' '.join(stop)


def preprocess(body):
	doc = nlp(body)
	processed = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
	return ' '.join(processed)

# Calculating number of Global Positioning Entity in a text
def count_gpe(txt):
	return sum([1 for token in nlp(txt).ents if token.label_ == 'GPE'])

# Claculating Number of Organisation in a Text
def count_org(txt):
	return sum([1 for token in nlp(txt).ents if token.label_ == 'ORG'])

# Calculating Number of Sentence in a text
def count_sentence(txt):
	doc = nlp(txt)
	return len([sent.text for sent in doc.sents])

# Extract only Noun and Proper Noun
def extract_noun(body):
	doc = nlp(body)
	cleaned_doc = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and (token.pos_ == 'NOUN' or token.pos_ == 'PROPN')]
	return ' '.join(cleaned_doc)
	
# Adding Pos Tag with the corresponding words
def spacy_pos(body):
	doc = nlp(body)
	cleaned_doc = [token.lemma_ + '_' + token.pos_ for token in doc if not token.is_stop and not token.is_punct]
	return ' '.join(cleaned_doc)

# Normalize the Custon Features
def normalize(df):
	df['sentence_count'] /= df['sentence_count'].max()
	df['gpe_count'] /= df['gpe_count'].max()
	df['org_count'] /= df['org_count'].max()
	
	return df


if __name__ == "__main__":
	args = create_arg_parser()
	
	test_df['body'] = np.vectorize(remove_url)(test_df['body'])
	
	test_df['body'] = np.vectorize(remove_newline)(test_df['body'])
	
	test_df['processed'] = np.vectorize(preprocess)(test_df['body'])
	
	start = t.time()
	
	test_df['gpe_count'] = [sum([1 for token in nlp(txt).ents if token.label_ == 'GPE']) for txt in test_df['body']]
	
	stop = t.time()
	print("\n Count GPE Time for test set: {}".format(stop - start))
	
	start = t.time()
	
	test_df['org_count'] = [sum([1 for token in nlp(txt).ents if token.label_ == 'ORG']) for txt in test_df['body']]
	
	stop = t.time()
	print("\n Count Name_entity Time for test set: {}".format(stop - start))
	
	start = t.time()
	
	test_df['sentence_count'] = [len([sent.text for sent in nlp(body).sents]) for body in test_df['body']]
	
	stop = t.time()
	print("\n Count Sentence for test set: {}".format(stop - start))
	
	
	start = t.time()
		   
	test_df['pos_tagged'] = [' '.join([token.lemma_ + '_' + token.pos_ for token in nlp(body) if not token.is_stop and not token.is_punct]) for body in test_df['processed']]
	
	stop = t.time()
	print("\n Adding Pos Tag to test set: {}".format(stop - start))
	
	start = t.time()
	
	test_df['noun'] = [' '.join([token.lemma_ for token in nlp(body) if not token.is_stop and not token.is_punct and (token.pos_ == 'NOUN' or token.pos_ == 'PROPN')]) for body in test_df['processed']]
	
	stop = t.time()
	print("\n Noun and Proper Noun Extraction for test set: {}".format(stop - start))

	test_df = normalize(test_df)
	
	test_df.to_csv('./processed_data/processed_custom_test.csv', index=False)
