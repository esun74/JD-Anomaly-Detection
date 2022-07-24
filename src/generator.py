import pandas as pd
import pickle
from transformers import BertTokenizer, BertForMaskedLM
import src.steps.mask
import src.steps.guess

class Generator():
	def __init__(self, method='RANDOM'):

		self.masks = src.steps.mask.create_masks(100, 512, 0.05)

		with open('./dataset/checkpoint.pickle', 'rb') as f:
			self.dataset = pickle.load(f)

		self.mask()

		if method == 'BERT':
			self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
			self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
			self.guess()

		elif method == 'RANDOM':
			self.wordset()
			self.wordinsert()

		with open('./src/generated.pickle', 'wb') as f:
			pickle.dump(self.dataset, f)

		with open('./src/masks.pickle', 'wb') as f:
			pickle.dump(self.masks, f)
		
	def mask(self):
		for ds in self.dataset:
			self.dataset[ds] = src.steps.mask.apply_mask(self.dataset[ds], self.masks)

	def guess(self):
		for ds in self.dataset:
			self.dataset[ds] = src.steps.guess.fill_mask(self.dataset[ds], self.model, self.tokenizer)

	def wordset(self):
		self.wordset = {} # dataset: [{train set of words not in label 0}, {1}, {2}...], [{test set of words not in label 0}, {1}, {2}...]
		for ds in self.dataset:
			self.wordset[ds] = src.steps.guess.get_words(self.dataset[ds])

	def wordinsert(self):
		for ds in self.dataset:
			self.dataset[ds] = src.steps.guess.random_fill(self.dataset[ds], self.wordset[ds])
