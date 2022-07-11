import os
import pandas as pd
import numpy as np
import sentencepiece
import transformers
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import tensorflow as tf
import random
import time

class Summarizer:
	def __init__(self, model, file):
		with tf.device('/cpu:0'):
			self.model = TFT5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base")
			self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
		self.dataset = pd.read_csv(file, index_col=0)[:5]
		self.dataset['Description Length'] = self.dataset['Description'].apply(self.num_words)

		try:

			self.fluff_indicators = set()
			with open('src/fluff_keywords.txt', 'r') as f:
				for l in f:
					self.fluff_indicators.add(l.strip())

			self.substance_indicators = set()
			with open('src/substance_keywords.txt', 'r') as f:
				for l in f:
					self.substance_indicators.add(l.strip())

		except FileNotFoundError as e:
			print(e)
			print('Your current working directory is:', os.getcwd())
			print('It should instead be the top level folder in the project.')

		self.clean_record = {}
		self.dataset['Cleaned Description'] = self.dataset.apply(self.clean, axis=1)
		self.dataset['Cleaned Description Length'] = self.dataset['Cleaned Description'].apply(self.num_words)
		self.dataset['Reduction'] = self.dataset[[
			'Description Length', 
			'Cleaned Description Length'
		]].apply(lambda x: (1 - (x[1] / x[0])) * 100, axis=1)

		self.dataset['Summary'] = self.dataset['Cleaned Description'].apply(self.summarize)

	@staticmethod
	def num_words(article):
		return len(article.split())

	def clean(self, item):

		article = item[2]

		# Lowercase everything
		article = article.lower()

		# Remove symbols
		excl = {
			'\\', '/', 
			'[', ']', 
			'(', ')', 
			'<', '>', 
			':', '_', ', ',	'"', '“', '”', '-', ' – ', '&', '’', '~'
			'a.m.', 'p.m.', 
			' n. ', ' s. ', ' e. ', ' w. ',
			' st. ', ' ave. ', ' blvd. ', 
			', etc.', 'i.e.', 'e.g.'
		}
		repl = {
			'1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.',
			' a ', ' b ', ' c ', ' d ', ' e ', ' f ', ' g ', ' h ', ' i ', ' j ', 
			'?', '!', ';', '•',
			'…', '...', '..',
		}
		for e in excl:
			article = article.replace(e, ' ')
		for r in repl:
			article = article.replace(r, '.')

		# Abbereviations that are actually relevant
		article = article.replace('b.s.', 'bachelors of science')
		article = article.replace('m.s.', 'masters of science')
		article = article.replace('u.s.', 'united states')

		# Remove fluff sentences via keywords
		substance, fluff = [], []
		for sentence in article.split('. '):

			# space out periods immediately followed by characters
			sentence = ' '.join(sentence.replace('.', ' ').split())

			# Remove duplicate sentences
			if not (sentence in substance or sentence in fluff) and len(sentence) > 5:
				words = set(sentence.split())
				if words & self.fluff_indicators and not words & self.substance_indicators:
					fluff.append(sentence)
				else:
					substance.append(sentence)
		article = '. '.join(substance)

		self.clean_record[item.name] = (substance, fluff)

		return article

	def summarize(self, article):

		print('#' * 100)
		print(article)
		print()
		tokenized_article = self.tokenizer(
			'summarize: ' + article,
			max_length=1024,
			truncation=True,
			return_tensors='tf',
		)
		summary = self.model.generate(
			tokenized_article['input_ids'],
			num_beams=4,
			no_repeat_ngram_size=2,
			min_length=10,
			max_length=300,
			early_stopping=True,
		)

		response = self.tokenizer.decode(summary[0], skip_special_tokens=True)

		print(response)

		return response

if __name__ == "__main__":

	s = Summarizer(
		model='google/t5-v1_1-base',
		file='.\\dataset\\Job-Recommendation-Engine-master\\Results\\JobsDataset.csv',
	)

	# Looking at the longest description

	longest = np.argmax(s.dataset['Cleaned Description Length'])
	print('Average Reduction: {0:.2f}%'.format(s.dataset['Reduction'].mean()))
	print('#' * 50, 'Longest Description', '#' * 50)
	print(
		'Longest Description:', s.dataset.iat[longest, 5], 
		'after {0:.2f}% reduction'.format(s.dataset.iat[longest, 6]), 
		'from', s.dataset.iat[longest, 3]
	)
	print('Substance', '-' * 100)
	for sentence in s.clean_record[longest + 1][0]:
		print(sentence)

	print('Fluff', '-' * 100)
	for sentence in s.clean_record[longest + 1][1]:
		print(sentence)

	# Looking at shortest descriptions

	for row in s.dataset.loc[s.dataset['Cleaned Description Length'].between(1, 5)].itertuples():
		print('#' * 50, 'Short Description', '#' * 50)
		print()
		print('Original:')
		print(row[3])
		print()
		print(row[6], 'after {0:.2f}% reduction'.format(row[7]), 'from', row[4])
		print('Substance', '-' * 100)
		for sentence in s.clean_record[row[0]][0]:
			print(sentence)
	
		print('Fluff', '-' * 100)
		for sentence in s.clean_record[row[0]][1]:
			print(sentence)

	# for row in s.dataset[['Cleaned Description', 'Summary']]:
	# 	print(row)
