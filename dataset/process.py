import pandas as pd
import numpy as np
import pickle
import dataset.steps.obtain
import dataset.steps.structure
import dataset.steps.clean

class Datasets:
	def __init__(self, refresh=True):

		if refresh:
			self.obtain()
			self.structure()
			self.clean()
			self.quicksave()
		else:
			try:
				self.quickload()
			except Exception as e:
				print(e)

	def obtain(self):
		self.dataset = {
			'20newsgroups': dataset.steps.obtain.newsgroups(),
			'agnews': dataset.steps.obtain.agnews(),
			'jobdescriptions': dataset.steps.obtain.jobdescriptions(),
		}

	def structure(self):
		self.dataset = {
			'20newsgroups': dataset.steps.structure.newsgroups(self.dataset['20newsgroups']),
			'agnews': dataset.steps.structure.agnews(self.dataset['agnews']),
			'jobdescriptions': dataset.steps.structure.jobdescriptions(self.dataset['jobdescriptions']),
		}

	def clean(self):
		for ds in self.dataset:
			self.dataset[ds] = dataset.steps.clean.structured_data(self.dataset[ds])

	def quicksave(self):
		with open('./dataset/checkpoint.pickle', 'wb') as f:
			pickle.dump(self.dataset, f)

	def quickload(self):
		with open('./dataset/checkpoint.pickle', 'rb') as f:
			self.dataset = pickle.load(f)
