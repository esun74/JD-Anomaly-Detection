import torch
import random

def fill_mask(data, model, tokenizer):

	def process(input):

		output = input

		def guess_sentence(article):

			tokenized = tokenizer(article, return_tensors='pt', max_length=512, truncation=True)
			tokenized_length = tokenized.input_ids.size()[1]
			predictions = model(**tokenized)[0]

			predicted_index = [torch.argmax(predictions[0, i]).item() for i in range(0, tokenized_length)]
			predicted_token = [tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in range(1, tokenized_length)]
			
			return tokenizer.convert_tokens_to_string(predicted_token)

		output['X2'] = output['X'].apply(guess_sentence)

		print(output)

		return output

	return process(data[0]), process(data[1])

def get_words(data):

	def process(input):

		output = [set() for _ in input['y'].unique()]
		for row in input.itertuples():
			for word in row.X.split():
				output[row.y].add(word)

		all_words = set().union(*output)
		for i in range(len(output)):
			output[i] = tuple(all_words.difference(output[i]))

		return output

	return process(data[0]), process(data[1])

def random_fill(data, words):

	def process(input, sample_words):

		output = input

		def inner(item):
			sentence = item.X.split()
			for i, w in enumerate(sentence):
				if w == '[MASK]':
					sentence[i] = random.choice(sample_words[item.y])
			return ' '.join(sentence)

		output['X'] = output.apply(inner, axis=1).values
		return output

	return process(data[0], words[0]), process(data[1], words[1])