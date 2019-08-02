""" This program uses the results of
the IBM Model1 implementation done by hand
to extract phrases from the given dataset and 
score their probabilities.
"""

from ibm_model1 import SentencePair
from ibm_model1 import IBMModel1
from nltk.translate.phrase_based import phrase_extraction

def phrase_based_extraction(data_file, iterations):
	""" Runs IBMModel 1 to obtain alignments, performs
	phrase based extraction and scores phrases.
	"""

	#: Running IBM Model 1.
	print("Running IBM Model.")
	sentence_pairs = IBMModel1(data_file, iterations)
	print("IBM Model training done. Now extracting phrases.")

	#: Extracting phrases from all sentence pairs.
	phrases = set()
	for pair in sentence_pairs:
		srctext = " ".join(pair.l1_sentence)
		trgtext = " ".join(pair.l2_sentence)
		alignment = list(pair.alignment())
		phrases.update(phrase_extraction(srctext, trgtext, alignment))
	phrases = list(phrases)

	#: Calculating t(L2|L1) for all phrase pairs p(L2, L1)
	#: Using the formula t(L2|L1) = count(L2, L1)/count(L1)
	phrase_p = [0] * len(phrases)
	for i, phrase in enumerate(phrases):
		num = 0
		denom = 0
		for pair in sentence_pairs:
			if phrase[2] in " ".join(pair.l1_sentence):
				denom += 1
				if phrase[3] in " ".join(pair.l2_sentence):
					num += 1
		phrase_p[i] = num/denom

	#: Displaying phrases in descending order of probability.
	phrases = [x for _, x in sorted(zip(phrase_p, phrases), reverse=True)]
	phrase_p.sort(reverse=True)
	for i in range(len(phrases)):
		print(phrases[i], phrase_p[i])

if __name__ == '__main__':

	phrase_based_extraction(
		data_file="data2.json",
		iterations=5)

