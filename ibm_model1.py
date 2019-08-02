""" This program implements IBM Model 1.
"""
import json
import itertools
import sys

class SentencePair:
	""" 
	This class represents sentence pairs. L1 represents the
	source language. L2 represents the target language. All
	possible alignment permutations are stored as well as their
	respective probabilities.
	"""
	def __init__(self, l1_sentence, l2_sentence):
		"""
		:param l1_sentence: source language sentence string
		:param l2_sentence: target language sentence string
		"""
		self.l1_sentence = l1_sentence.split()
		self.l2_sentence = l2_sentence.split()
		self.set_alignments()

	def set_alignments(self):
		""" Method creates and stores all possible alignment
		permutations for this sentence pair. This mehtod is 
		called once at initialization.
		"""

		#: First, each word in L1 is generated a list of
		#: tuples between its index and all possible indices
		#: in L2. E.g: For the word at index 0 in L1, the list
		#: [(0,0), (0,1)] is generated, assuming that there
		#: are only two words in L2. 'arg' contains all such
		#: lists.
		arg = []
		for l1_index in range(len(self.l1_sentence)):
			tuple_list = []
			for l2_index in range(len(self.l2_sentence)):
				tuple_list.append((l1_index, l2_index))
			arg.append(tuple_list)

		#: Now, alignments are created by forming all
		#: combinations of tuples. Each alignment 
		#: created will contain 1 tuple from each of the
		#: in 'arg'. NOTE: Each alignment is a tuple of tuples.
		self.alignments = list(itertools.product(*arg))

		#: Corresponding array of alignment probabilities.
		#: P(self.alignments[i]) = self.alignment_p[i]
		self.alignment_p = [1] * len(self.alignments)

	def print_all_alignments(self):
		""" Utility method to print sentence alignments
		and their respective probabilities.
		"""
		for i in range(len(self.alignments)):
			print(self.alignments[i], "  :  ", self.alignment_p[i])

	def print(self):
		""" Utility method to print sentence pair and
		its likeliest alignment.
		"""
		print("* Sentence Pair *")
		print("\t", self.l1_sentence)
		print("\t", self.l2_sentence)
		print("\t", self.alignment())

	def alignment(self):
		""" Returns likeliest alignment of this
		sentence pair.
		"""
		max_p = max(self.alignment_p)
		max_index = self.alignment_p.index(max_p)
		return self.alignments[max_index]


def print_tp_table(tp_table):
	""" Utility method to print the translation prob.
	table.
	"""
	print("Printing tp table\n")
	for l2_word in tp_table:
		print(l2_word, " ", tp_table[l2_word], "\n")

def IBMModel1(data_file, iterations):
	""" IBM Model 1 implementation.

	:param datafile: Input data in the form
		of a json file.
	:param iterations: Number of iterations
		for the model to run.
	"""
	#: Loading training corpus sentences.
	with open(data_file) as training_corpus:
		input_sentences = json.load(training_corpus)

	#: Creating SentencePair objects from json file.
	sentence_pairs = []
	for pair in input_sentences:
		sentence_pairs.append(SentencePair(pair["fr"], pair["en"]))

	#: Gather sets of unique words in L1, and L2
	l1_words = set()
	l2_words = set()
	for pair in sentence_pairs:
		l1_words.update(set(pair.l1_sentence))
		l2_words.update(set(pair.l2_sentence))
	#: Converting sets to lists in order to preserve
	#: element order.
	l1_words = list(l1_words)
	l2_words = list(l2_words)

	#: Creating a translation probability table between
	#: the words in L1 and L2 in the form of a dict of
	#: dicts and initializing it with uniform probability.
	#: E.g:
	#:
	#:	| **	| le 	| bleu 	| rouge | -> L1 words
	#: 	| the 	| 1/3	| 1/3	| 1/3	|
	#:	| blue	| 1/3	| 1/3 	| 1/3	|
	#: 	| red 	| 1/3 	| 1/3	| 1/3 	|
	#: 		|__ L2 words
	#: 
	#: P(L1 word | L2 word) = tp_table[L2 word][L1 word]

	tp_table = {}
	uniform_p = 1/len(l1_words)
	for l2_word in l2_words:
		tp_table[l2_word] = {}
		for l1_word in l1_words:
			tp_table[l2_word][l1_word] = uniform_p

	#: EM Algorithm begins.
	for iteration in range(iterations):

		#: STEP 1: Updating the P(a, L1| L2) values,
		#: alignment probability values.

		#: Iterating through all alignments.
		for pair in sentence_pairs:
			total_p = 0
			for i in range(len(pair.alignments)):
				pair.alignment_p[i] = 1
				for (l1_index, l2_index) in pair.alignments[i]:
					l1_word = pair.l1_sentence[l1_index]
					l2_word = pair.l2_sentence[l2_index]
					pair.alignment_p[i] *= tp_table[l2_word][l1_word]
				total_p += pair.alignment_p[i]

			#: Normalizing the alignment prob. values.
			for i in range(len(pair.alignments)):
				if pair.alignment_p[i] == 0 and total_p == 0:
					continue
				pair.alignment_p[i] /= total_p

		#: STEP 2: Updating the translation prob. table.
		for i, l2_word in enumerate(l2_words):
			total_p = 0
			for j, l1_word in enumerate(l1_words):
				tp_table[l2_word][l1_word] = 0
				for pair in sentence_pairs:
					for index, alignment in enumerate(pair.alignments):
						if l1_word in pair.l1_sentence and l2_word in pair.l2_sentence:
							l1_index = pair.l1_sentence.index(l1_word)
							l2_index = pair.l2_sentence.index(l2_word)
							if (l1_index, l2_index) in alignment:
								tp_table[l2_word][l1_word] += pair.alignment_p[index]
				total_p += tp_table[l2_word][l1_word]

			#: Normalizing the fractional count
			for l1_word in l1_words:
				if tp_table[l2_word][l1_word] == 0 and total_p == 0:
					continue
				tp_table[l2_word][l1_word] /= total_p

		# for pair in sentence_pairs:
		# 	pair.print() 

	return sentence_pairs

if __name__ == '__main__':
	""" E.g usage: `python ibm_model1.py <data_file> <number of iterations>`
	"""
	sentence_pairs = IBMModel1(data_file=sys.argv[1], iterations=int(sys.argv[2]))

	#: Printing the sentence pairs, and alignments obtained
	for pair in sentence_pairs:
		pair.print()

		















