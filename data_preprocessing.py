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

#from read_json import train_df, val_df, test_df

train_df = pd.read_csv('./data/train.csv')
val_df = pd.read_csv('./data/val.csv')
test_df = pd.read_csv('./data/test.csv')


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


train_df['news_key'] = np.vectorize(get_key)(train_df['newspaper_name'])

val_df['news_key'] = np.vectorize(get_key)(val_df['newspaper_name'])

test_df['news_key'] = np.vectorize(get_key)(test_df['newspaper_name'])

train_df['political_orientation'] = np.vectorize(get_orientation)(train_df['newspaper_name'])

val_df['political_orientation'] = np.vectorize(get_orientation)(val_df['newspaper_name'])

test_df['political_orientation'] = np.vectorize(get_orientation)(test_df['newspaper_name'])


# Downsampling and Upsampling
def up_down_sampling(train_df):
	df_aus =  resample(train_df[train_df['news_key']=='Australia'], replace=False, n_samples=4000)

	df_ny = resample(train_df[train_df['news_key']=='Newyork'], replace=False, n_samples=4000)
	
	df_was = resample(train_df[train_df['news_key']=='Washington'], replace=False, n_samples=3500)
	 
	df_her = resample(train_df[train_df['news_key']=='Herald'], replace=False, n_samples=3500)
	
	df_age = resample(train_df[train_df['news_key']=='Age'], replace=False, n_samples=3000)
	
	df_in = resample(train_df[train_df['news_key']=='India'], replace=True, n_samples=2500)
	
	df_gur = resample(train_df[train_df['news_key']=='Guardian'], replace=True, n_samples=500)
	
	df_hin = resample(train_df[train_df['news_key']=='Hindu'], replace=True, n_samples=500)
	
	df_tm = resample(train_df[train_df['news_key']=='Times'], replace=True, n_samples=250)
	
	df_train = pd.concat([df_aus, df_ny, df_was, df_her, df_age, df_gur, df_in, df_hin, df_tm])
	
	train_df = shuffle(df_train)
	
	return train_df


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


#Extract only Noun and Proper Noun
def extract_noun(body):
	doc = nlp(body)
	cleaned_doc = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and (token.pos_ == 'NOUN' or token.pos_ == 'PROPN')]
	return ' '.join(cleaned_doc)


# Adding Pos Tag with the corresponding words
def spacy_pos(body):
	doc = nlp(body)
	cleaned_doc = [token.lemma_ + '_' + token.pos_ for token in doc if not token.is_stop and not token.is_punct]
	return ' '.join(cleaned_doc)


#Normalize the Custon Features
def normalize(df):
	df['sentence_count'] /= df['sentence_count'].max()
	df['gpe_count'] /= df['gpe_count'].max()
	df['org_count'] /= df['org_count'].max()
	
	return df


if __name__ == "__main__":
	args = create_arg_parser()
	
	#Applying Up and Down Sampling in Train Set
	train_df = up_down_sampling(train_df)

	train_df['body'] = np.vectorize(remove_url)(train_df['body'])

	val_df['body'] = np.vectorize(remove_url)(val_df['body'])
	
	test_df['body'] = np.vectorize(remove_url)(test_df['body'])
	
	train_df['body'] = np.vectorize(remove_newline)(train_df['body'])
	
	val_df['body'] = np.vectorize(remove_newline)(val_df['body'])
	
	test_df['body'] = np.vectorize(remove_newline)(test_df['body'])
	
	train_df['processed'] = np.vectorize(preprocess)(train_df['body'])
	
	val_df['processed'] = np.vectorize(preprocess)(val_df['body'])
	
	test_df['processed'] = np.vectorize(preprocess)(test_df['body'])
	
	start = t.time()
	
	train_df['gpe_count'] = [sum([1 for token in nlp(txt).ents if token.label_ == 'GPE']) for txt in train_df['body']]
	
	val_df['gpe_count'] = [sum([1 for token in nlp(txt).ents if token.label_ == 'GPE']) for txt in val_df['body']]
	
	test_df['gpe_count'] = [sum([1 for token in nlp(txt).ents if token.label_ == 'GPE']) for txt in test_df['body']]
	
	stop = t.time()
	print("\n Count GPE Time: {}".format(stop - start))
	
	start = t.time()
	
	train_df['org_count'] = [sum([1 for token in nlp(txt).ents if token.label_ == 'ORG']) for txt in train_df['body']]
	
	val_df['org_count'] = [sum([1 for token in nlp(txt).ents if token.label_ == 'ORG']) for txt in val_df['body']]
	
	test_df['org_count'] = [sum([1 for token in nlp(txt).ents if token.label_ == 'ORG']) for txt in test_df['body']]
	
	stop = t.time()
	print("\n Count Name_entity Time: {}".format(stop - start))
	
	start = t.time()
	
	train_df['sentence_count'] = [len([sent.text for sent in nlp(body).sents]) for body in train_df['body']]
	
	val_df['sentence_count'] = [len([sent.text for sent in nlp(body).sents]) for body in val_df['body']]
	
	test_df['sentence_count'] = [len([sent.text for sent in nlp(body).sents]) for body in test_df['body']]
	
	stop = t.time()
	print("\n Count Sentence: {}".format(stop - start))
	
	start = t.time()
	
	train_df['pos_tagged'] = [' '.join([token.lemma_ + '_' + token.pos_ for token in nlp(body) if not token.is_stop and not token.is_punct]) for body in train_df['processed']]
	
	val_df['pos_tagged'] = [' '.join([token.lemma_ + '_' + token.pos_ for token in nlp(body) if not token.is_stop and not token.is_punct]) for body in val_df['processed']]
	
	test_df['pos_tagged'] = [' '.join([token.lemma_ + '_' + token.pos_ for token in nlp(body) if not token.is_stop and not token.is_punct]) for body in test_df['processed']]
	stop = t.time()
	
	print("\n Adding Pos Tag: {}".format(stop - start))
	#train_df['gpe_count'] = [sum(token.label_=='GPE' for token in nlp(text).ents) for text in train_df['body']]
	
	start = t.time()
	
	train_df['noun'] = [' '.join([token.lemma_ for token in nlp(body) if not token.is_stop and not token.is_punct and (token.pos_ == 'NOUN' or token.pos_ == 'PROPN')]) for body in train_df['processed']]
	
	val_df['noun'] = [' '.join([token.lemma_ for token in nlp(body) if not token.is_stop and not token.is_punct and (token.pos_ == 'NOUN' or token.pos_ == 'PROPN')]) for body in val_df['processed']]
	
	test_df['noun'] = [' '.join([token.lemma_ for token in nlp(body) if not token.is_stop and not token.is_punct and (token.pos_ == 'NOUN' or token.pos_ == 'PROPN')]) for body in test_df['processed']]
	
	stop = t.time()
	print("\n Noun and Proper Noun Extraction: {}".format(stop - start))
	
	train_df = normalize(train_df)
	val_df = normalize(val_df)
	test_df = normalize(test_df)
	
	train_df.to_csv('./processed_data/processed_train.csv', index=False)
	
	val_df.to_csv('./processed_data/processed_val.csv', index=False)
	
	test_df.to_csv('./processed_data/processed_test.csv', index=False)
