from spacy.lang.en import English
from spacy.tokens import Doc
from collections import Counter
import math
import random

nlp = English(pipeline=[],max_length=5000000)

def get_unigrams(doc,do_lower=True):
	tokens = [x.text for x in doc]
	if do_lower:
		return [x.lower() for x in tokens]
	else:
		return tokens

def get_bigrams(doc):
	unigrams = get_unigrams(doc)
	return zip(unigrams[:-1],unigrams[1:])

def get_trigrams(doc):
	unigrams = get_unigrams(doc)
	return zip(unigrams[:-2],unigrams[1:-1],unigrams[2:])
	return trigrams

def get_ngram_counts(doc):
	unigrams = Counter()
	bigrams = Counter()
	trigrams = Counter()
	unigrams.update(get_unigrams(doc))
	bigrams.update(get_bigrams(doc))
	trigrams.update(get_trigrams(doc))
	return unigrams, bigrams, trigrams

def calc_ngram_prob(tri,bigrams,trigrams):
	tri_count = trigrams[tri] if tri in trigrams else 0
	context = (tri[0],tri[1])
	bi_count = bigrams[context] if context in bigrams else 0
	return math.log(tri_count/bi_count) if tri_count != 0 else -math.inf

def get_possible_next_words(prev,bigrams,trigrams):
	penultimate,last = prev.split(' ')[-2:]
	return [(t,calc_ngram_prob(t,bigrams,trigrams)) for t in trigrams if t[0] == penultimate and t[1] == last]

def predict_next_word(tri,bigrams,trigrams):
	possibilities = get_possible_next_words(tri,bigrams,trigrams)
	best = sorted(possibilities,key=lambda x:x[1])[-1]
	return best[0][2]

def sample_next_word(tri,bigrams,trigrams):
	possibilities = get_possible_next_words(tri,bigrams,trigrams)
	best = random.choices([p[0] for p in possibilities],weights=[math.exp(p[1]) for p in possibilities],k=1)
	return best[0][2]

def generate_text(context,n,bigrams,trigrams,mode="top"):
	sampler = sample_next_word if mode == "random" else predict_next_word
	for i in range(n):
		context = context + ' ' + sampler(context,bigrams,trigrams)
	return context

def make_doc(files):
	docs = []
	for f in files:
		with open(f,'r',encoding="latin1") as fn:
			docs.append(nlp(fn.read()))
	return Doc.from_docs(docs)

def main():
	emma = "gutenberg_data/austen-emma.txt"
	persuasion = "gutenberg_data/austen-persuasion.txt"
	training_doc = make_doc([emma,persuasion])
	unigrams, bigrams, trigrams = get_ngram_counts(training_doc)
	calc_ngram_prob(("an","agreeable","surprize"),bigrams,trigrams)
	get_possible_next_words("an agreeable",bigrams,trigrams)
	predict_next_word("an agreeable",bigrams,trigrams)
	print(generate_text("an agreeable",5,bigrams,trigrams))
	print(generate_text("an agreeable",5,bigrams,trigrams,mode="random"))

main()