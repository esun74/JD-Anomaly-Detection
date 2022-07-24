import pandas as pd
import pickle

class Discriminator():
	def __init__(self, method='RANDOM'):

		with open('./src/generated.pickle', 'rb') as f:
			self.dataset = pickle.load(f)

		with open('./src/masks.pickle', 'rb') as f:
			self.masks = pickle.load(f)

		print()
		for ds in self.dataset:
			print('Train:')
			print(self.dataset[ds][0])
			print('Test:')
			print(self.dataset[ds][1])
		print(self.masks.shape)
