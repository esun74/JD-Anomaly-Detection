from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import tensorflow_datasets as tfds

def newsgroups():

	train = pd.DataFrame(fetch_20newsgroups(
		data_home='./dataset/20newsgroups/', 
		subset='train', 
		shuffle=True,
		remove=('headers', 'footers', 'quotes'),
		return_X_y=True,
	))

	test = pd.DataFrame(fetch_20newsgroups(
		data_home='./dataset/20newsgroups/', 
		subset='test', 
		shuffle=True,
		remove=('headers', 'footers', 'quotes'),
		return_X_y=True,
	))

	return train, test

def agnews():

	train = tfds.as_dataframe(tfds.load(
		'ag_news_subset', 
		split='train',
		shuffle_files=True,
		data_dir='./dataset/agnews/'
	))

	test = tfds.as_dataframe(tfds.load(
		'ag_news_subset', 
		split='test',
		shuffle_files=True,
		data_dir='./dataset/agnews/'
	))

	return train, test

def jobdescriptions():

	dataset = pd.read_csv('./dataset/Job-Recommendation-Engine-master/Results/JobsDataset.csv', index_col=0)
	test = dataset.sample(int(len(dataset) * 0.1))
	train = dataset.drop(test.index)

	return train, test
