import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import torch
import random
import string
import nltk
nltk.download('stopwords')

class Summarizer:
	def __init__(self, file, fluff, substance, trunc_at=480, masks=100):
		# with tf.device('/cpu:0'):

		# BERT with language modeling head (pytorch)
		self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.dataset = pd.read_csv(file, index_col=0)[:100]
		self.dataset['Description Length'] = self.dataset['Description'].apply(self.num_words)
		self.stopwords = set(nltk.corpus.stopwords.words('english'))
		self.mask = np.zeros((trunc_at, masks))
		self.trunc_at = trunc_at

		print(self.mask)

		# self.fluff_indicators = set()
		# with open(fluff, 'r') as f:
		# 	for l in f:
		# 		self.fluff_indicators.add(l.strip())

		# self.substance_indicators = set()
		# with open(substance, 'r') as f:
		# 	for l in f:
		# 		self.substance_indicators.add(l.strip())

		# self.clean_record = {}
		# self.dataset['Cleaned Description'] = self.dataset.apply(self.clean, axis=1).values
		# self.dataset['Cleaned Description Length'] = self.dataset['Cleaned Description'].apply(self.num_words).values
		# self.dataset['Reduction'] = self.dataset[[
		# 	'Description Length', 
		# 	'Cleaned Description Length'
		# ]].apply(lambda x: (1 - (x[1] / x[0])) * 100, axis=1).values

		# self.dataset = self.dataset.loc[self.dataset['Cleaned Description Length'] > 25]

		# self.dataset['Mask Filter'] = self.dataset.apply(self.create_mask, axis=1).values
		# self.dataset['Masked Description'] = self.dataset.apply(self.insert_mask, axis=1).values
		# self.dataset['Filled Description'] = self.dataset.apply(self.fill_mask, axis=1).values

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

		# Remove punctuation 
		article = article.translate(str.maketrans('', '', string.punctuation))

		# Remove stopwords
		article = ' '.join([w for w in nltk.tokenize.word_tokenize(article) if w not in self.stopwords])

		# Trim, saving 32 tokens for words split into multiple token parts e.g. "eating" -> "eat", "**ing"
		article = ' '.join(article.split()[:480])

		return article

	@staticmethod
	def create_mask(item):

		# 5% contamination
		article = item[4].split()
		num_masks = max(1, int(len(article) * 0.05))
		masking = [0 for _ in article]
		for m in random.choices(range(len(masking)), k=num_masks):
			masking[m] = 1
		return masking

	@staticmethod
	def insert_mask(item):
		article = item[4].split()
		masking = item[7]
		return ' '.join(['[MASK]' if masking[i] else w for i, w in enumerate(article)])

	def fill_mask(self, item):
		article = item[8]

		tokenized = self.tokenizer(article, return_tensors='pt')
		tokenized_length = tokenized.input_ids.size()[1]
		predictions = self.model(**tokenized)[0]

		predicted_index = [torch.argmax(predictions[0, i]).item() for i in range(0, tokenized_length)]
		predicted_token = [self.tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in range(1, tokenized_length)]
		
		filled_article = self.tokenizer.convert_tokens_to_string(predicted_token)

		return filled_article

if __name__ == "__main__":

	s = Summarizer(
		file='./dataset/Job-Recommendation-Engine-master/Results/JobsDataset.csv',
		fluff='src/fluff_keywords.txt',
		substance='src/substance_keywords.txt',
	)

	# r = random.randrange(len(s.dataset))
	# for i, c in enumerate(s.dataset.columns):
	# 	print(c)
	# 	print(s.dataset.iloc[r, i])
	# 	print()
