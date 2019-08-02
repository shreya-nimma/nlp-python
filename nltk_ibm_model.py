""" This program runs nltk implementations of IBM Models 1 and 2
on a given dataset.
"""
from nltk.translate import AlignedSent
from nltk.translate import IBMModel1, IBMModel2
import json, sys

def test_model(model_number, dataset_file, iterations):

	Model = IBMModel1 if model_number == 1 else IBMModel2

	bitext = []

	#: Opening source file.
	with open(dataset_file) as input_text:
		data = json.load(input_text)

	#: Encapsulating sentence pairs as AlignedSents.
	for pair in data:
		sentence_1 = pair["fr"].split()
		sentence_2 = pair["en"].split()
		bitext.append(AlignedSent(sentence_1, sentence_2))

	ibm1 = Model(bitext, iterations)

	#: Printing sentence pairs and obtained
	#: alignments.
	for test_sentence in bitext:
		print(test_sentence.words)
		print(test_sentence.mots)
		print(test_sentence.alignment)

if __name__ == '__main__':
	""" E.g usage:
	` python nltk_ibm_model.py <model_number> <dataset_file> `
	"""

	model = int(sys.argv[1])
	dataset = sys.argv[2]
	iterations = int(sys.argv[3])

	test_model(model_number=model, dataset_file=dataset, iterations=iterations)