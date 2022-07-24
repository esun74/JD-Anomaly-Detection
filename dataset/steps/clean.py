import pandas as pd
import numpy as np
import nltk
import string
import re

# Global variables

minimum_word_count = 25

fluff_indicators = set()
with open('./dataset/steps/fluff_keywords.txt', 'r') as f:
	for l in f:
		fluff_indicators.add(l.strip())

substance_indicators = set()
with open('./dataset/steps/substance_keywords.txt', 'r') as f:
	for l in f:
		substance_indicators.add(l.strip())

nltk.download('stopwords', download_dir='./dataset/stopwords/', quiet=True)
stopwords = set(nltk.corpus.stopwords.words('english'))

def structured_data(data):

	def process(input):

		output = input.copy()
		output.loc[:, 'X'] = output['X'].map(clean).values
		output = output.loc[output['X'].apply(lambda x: len(x.split())).values > minimum_word_count]

		return output

	return process(data[0]), process(data[1])

def clean(article):

	# Lowercase everything
	article = article.lower()

	# Abbereviations that are actually relevant
	article = article.replace('b.s.', 'bachelors of science')
	article = article.replace('b.a.', 'bachelors of art')
	article = article.replace('m.s.', 'masters of science')
	article = article.replace('m.a.', 'masters of art')
	article = article.replace(' 1 ', ' one ')
	article = article.replace(' 2 ', ' two ')
	article = article.replace(' 3 ', ' three ')
	article = article.replace(' 4 ', ' four ')
	article = article.replace(' 5 ', ' five ')
	article = article.replace(' 6 ', ' six ')
	article = article.replace(' 7 ', ' seven ')
	article = article.replace(' 8 ', ' eight ')
	article = article.replace(' 9 ', ' nine ')
	article = article.replace(' 0 ', ' zero ')

	# Remove symbols
	
	excl = {
		'a.m.', 'p.m.', 
		' n. ', ' s. ', ' e. ', ' w. ',
		' st. ', ' ave. ', ' blvd. ', 
		', etc.', 'i.e.', 'e.g.',
		'1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.',
		' a ', ' b ', ' c ', ' d ', ' e ', ' f ', ' g ', ' h ', ' i ', ' j ', '_'
	}
	for e in excl:
		article = article.replace(e, ' ')

	# Add Extractive Summarization here in the future

	# Remove all non alphanumeric characters 
	article = re.sub(r'[\W\d]+\ *', ' ', article)

	# Remove 1-2 character words
	article = ' '.join([w for w in article.split() if len(w) > 2])

	# Remove stopwords
	article = ' '.join([w for w in nltk.tokenize.word_tokenize(article) if w not in stopwords][:512])

	return article
