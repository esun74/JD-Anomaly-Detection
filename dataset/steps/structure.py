import numpy as np
import pandas as pd

def newsgroups(data):

	# 0: Computers
	# 1: Recreation (sports)
	# 2: Science
	# 3: Misc. 
	# 4: Politics
	# 5: Religion
	grouping = {
		0: 5, # alt.atheism
		1: 0, # comp.graphics
		2: 0, # comp.os.ms-windows.misc
		3: 0, # comp.sys.ibm.pc.hardware
		4: 0, # comp.sys.mac.hardware
		5: 0, # comp.windows.x
		6: 3,  # misc.forsale
		7: 1,  # rec.autos
		8: 1,  # rec.motorcycles
		9: 1,  # rec.sport.baseball
		10: 1,  # rec.sport.hockey
		11: 2,  # sci.crypt
		12: 2,  # sci.electronics
		13: 2,  # sci.med
		14: 2,  # sci.space
		15: 5,  # soc.religion.christian
		16: 4,  # talk.politics.guns
		17: 4,  # talk.politics.mideast
		18: 4,  # talk.politics.misc
		19: 5,  # talk.religion.misc
	}

	def process(input):

		output = pd.DataFrame(input.T)
		output.columns = ['X', 'y']
		output.y = group(output.y, grouping)
		return output

	return process(data[0]), process(data[1])

def agnews(data):

	# 0: World
	# 1: Sports
	# 2: Business
	# 3: Sci/Tech

	def process(input):

		output = input
		output['description'] = output['title'] + b': ' + output['description']
		output['description'] = output['description'].str.decode('utf-8')
		output = output[['description', 'label']]
		output.columns = ['X', 'y']

		return output

	return process(data[0]), process(data[1])

def jobdescriptions(data):

	# 0: Business Intelligence
	# 1: Data Analyst
	# 2: Data Scientist / MLE
	# 3: Data Engineer
	# 4: Other
	grouping = {
		'Data Scientist': 1,
		'Data Analyst': 0,
		'Data Architect': 2,
		'Data Engineer': 3,
		'Statistics': 1,
		'Database Administrator': 3,
		'Business Analyst': 1,
		'Data and Analytics Manager': 1,
		'Machine Learning': 2,
		'Artificial Intelligence': 0,
		'Deep Learning': 2,
		'Business Intelligence Analyst': 0,
		'Data Visualization Expert': 0,
		'Data Quality Manager': 3,
		'Big Data Engineer': 3,
		'Data Warehousing': 3,
		'Technology Integration': 4,
		'IT Consultant': 4,
		'IT Systems Administrator': 4,
		'Cloud Architect': 2,
		'Technical Operations': 4,
		'Cloud Services Developer': 4,
		'Full Stack Developer': 4,
		'Information Security Analyst': 1,
		'Network Architect': 3,
	}

	def process(input):

		output = input

		output['Query'] = group(output['Query'], grouping)
		output['Description'] = output['Job Title'] + ': ' + output['Description']
		output = output[['Description', 'Query']]
		output.columns = ['X', 'y']
		return output

	return process(data[0]), process(data[1])

def group(input, grouping):
	u, i = np.unique(input, return_inverse=True)
	return np.array([grouping[x] for x in u])[i]
