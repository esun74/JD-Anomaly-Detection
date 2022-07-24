import numpy as np
import random

def create_masks(n, tokens, contamination):

	masks = np.zeros((n, tokens))

	for row in range(n):
		for col in random.sample(range(tokens), k=max(1, round(tokens * contamination))):
			masks[row, col] = 1

	return masks

def apply_mask(data, masks):

	def process(input):

		output = input

		output['mask'] = np.random.randint(0, masks.shape[0], len(output))

		def mask_up(item):
			current_mask = masks[item[1]]
			words = item[0].split()
			words = ['[MASK]' if current_mask[i] else w for i, w in enumerate(words)]
			return ' '.join(words)

		output['X'] = output[['X', 'mask']].apply(mask_up, axis=1)

		return output

	return process(data[0]), process(data[1])
