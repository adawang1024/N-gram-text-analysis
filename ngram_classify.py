#!/usr/bin/env python3
import os
import argparse
import html
from lxml import etree
from collections import Counter
import math
from ngram_generate import make_doc,get_ngram_counts,calc_ngram_prob

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk

nltk.download('gutenberg')
from nltk.corpus import gutenberg

from spacy.lang.en import English
nlp = English(pipeline=[], max_length=5000000)


def read_one(fp): 
    cnt = Counter()
    with open(fp,'r', encoding='latin1') as file:
        text =  file.read()
        doc = nlp(text)
        for tokens in doc: 
            token_text = tokens.text.lower()
            if token_text != "\n": 
                cnt[token_text] += 1
    return cnt
        

def read_all(dirPath, extension=None): 
    cnt = Counter()
    for dirpath, dirnames,filenames in os.walk(dirPath): 
        for file in filenames: 
            if extension is None or os.path.splitext(file)[-1] == extension.lower(): 
                fp = os.path.join(dirpath,file)
                cnt += read_one(fp)
    return cnt
    


def do_xml_parse(fp, tag):
    """ 
    Iteratively parses XML files
    """
    fp.seek(0)

    for (event, elem) in etree.iterparse(fp, tag=tag):
        yield elem
        elem.clear()

def get_articles(args, attribute, value):
    unigram_cnt = Counter()
    bigram_cnt = Counter()
    trigram_cnt = Counter()
    for article in do_xml_parse(args.articles,'article'): 
        if article.get(attribute) == value: 
            all_text = html.unescape(' '.join(article.itertext())) 
            all_text = nlp(all_text)

            unigram = get_unigrams(all_text)
            unigram_cnt.update(unigram)

            bigram = get_bigrams(all_text)
            bigram_cnt.update(bigram)

            trigram = get_trigrams(all_text)

            trigram_cnt.update(trigram)

    return unigram_cnt,bigram_cnt,trigram_cnt

        
def get_unigrams(doc, do_lower=True): 
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

def compare(train, test, unique=False):
    not_in_train = 0 
    total = 0
    if unique: 
        for token in test: 
            if token not in train: 
                not_in_train += 1
        return not_in_train,len(test)
    else: 
        for token in test.elements(): 
            if token not in train: 
                not_in_train += 1
            total += 1
        return not_in_train,total
        


def calc_text_perplexity(doc, bigram_collection, trigram_collection):
    log_prob_sum = 0.0
    for t in get_trigrams(doc): 
        log_prob_sum += calc_ngram_prob(t,bigram_collection,trigram_collection)
    # Normalize by the length of the document
    doc_length = len(doc)
    normalized_log_prob_sum = log_prob_sum / doc_length
    # Calculate perplexity (exponentiate the negative log likelihood)
    perplexity = math.exp(-normalized_log_prob_sum)
    
    return perplexity

def add1_smoothing(doc, bigram_collection, trigram_collection): 
    for bigram in bigram_collection: 
        bigram_collection[bigram] += 1
    for trigram in trigram_collection:
        trigram_collection[trigram] += 1
    ngrams = get_ngram_counts(doc)
    for bigram in ngrams[1]: 
        if bigram not in bigram_collection: 
           bigram_collection[bigram] = 1
        
    for trigram in ngrams[2]: 
        if trigram not in trigram_collection: 
           trigram_collection[trigram] = 1




def do_experiment(args, attribute, train_value, test_value): 
    """Print a pandoc-compatible table of experiment results"""
    train = get_articles(args, attribute, train_value) #unigram, bigram and trigram counter
    test = get_articles(args, attribute, test_value)

    table_header = "Results for {}, using {} as train and {} as test:"
    print(table_header.format(attribute, train_value, test_value))

    print("| Order | Type/Token | Total | Zeros | % Zeros")
    print("| ----  | ------ | ----- | ----- | ")
    table_row = "| Unigram | {typetoken} | {total} | {zeros} | {pct:.1%} | "
    table_row_2 = "| Bigram | {typetoken} | {total} | {zeros} | {pct:.1%} | "
    table_row_3 = "| Trigram | {typetoken} | {total} | {zeros} | {pct:.1%} | "
    for do_types in (True, False):
        typetoken = "Type" if do_types else "Token" 
        num_zeros, N = compare(train[0], test[0], do_types)
        num_zeros_2, N2 = compare(train[1], test[1], do_types)
        num_zeros_3, N3 = compare(train[2], test[2], do_types)
        print(table_row.format(typetoken=typetoken, 
              total=N, zeros=num_zeros, pct=num_zeros/N))
        print(table_row_2.format(typetoken=typetoken, 
              total=N2, zeros=num_zeros_2, pct=num_zeros_2/N2))
        print(table_row_3.format(typetoken=typetoken, 
              total=N3, zeros=num_zeros_3, pct=num_zeros_3/N3))
        
    print()



def one_doc_frequency(doc): 
    token_len = [] 
    for token in doc: 
        token_len.append(len(token))
    data = pd.Series(token_len)
    ax = sns.distplot(data,color = "purple")
    ax.set(xlabel = 'token length',ylabel = "frequency")
    plt.xlim(0,20)
    plt.show()
    plt.show()

def word_frequency(dirPath): 
    text_len = []
    for text in gutenberg.fileids(): 
        text_len.append(len(gutenberg.words(text)))
    data = pd.Series(text_len)
    ax = sns.distplot(data,color = "red")
    ax.set(xlabel = 'token length',ylabel = "frequency")
    plt.xlim(0,1e6)
    plt.show()

def main(args):
    # res = get_articles(args,'hyperpartisan','true')
    # print(res[0]["the"])
    # print(res[0]["california"])
    # print(res[0]["zero"])

    # print(compare(Counter([1,2,3]), Counter([3,4,4]), unique=True))
    # print(compare(Counter([1,2,3]), Counter([3,4,4]), unique=False))


    # do_experiment(args,'hyperpartisan','false','true')
    # do_experiment(args,'hyperpartisan','true','false')
   
    # do_experiment(args,'randomchunk','a','b')
    # do_experiment(args,'randomchunk','b','a')

    # doc = read_one(args)
    # unigrams, bigrams, trigrams = get_ngram_counts(doc)
    # print(calc_text_perplexity(doc, bigrams, trigrams))
   
    # emma = "gutenberg_data/austen-emma.txt"
    # persuasion = "gutenberg_data/austen-persuasion.txt"
    # training_doc = make_doc([emma,persuasion])
    # unigrams, bigrams, trigrams = get_ngram_counts(training_doc)
    # # print(list(bigrams.items())[:5])
    # print(calc_text_perplexity(training_doc, bigrams, trigrams))

    # emma = "gutenberg_data/bible-kjv.txt"
    # training_doc = make_doc([emma])
    # unigrams, bigrams, trigrams = get_ngram_counts(training_doc)
    # # print(list(bigrams.items())[:5])
    # print(calc_text_perplexity(training_doc, bigrams, trigrams))

    # austen = "gutenberg_data/austen-sense.txt"
    # test_doc = make_doc([austen])
    # add1_smoothing(test_doc,bigrams,trigrams)
    # print(list(bigrams.items())[:5])
    # print(list(bigrams.items())[-5:])
    
    # emma = "gutenberg_data/austen-emma.txt"
    # persuasion = "gutenberg_data/austen-persuasion.txt"
    # training_doc = make_doc([emma,persuasion])
    # unigrams, bigrams, trigrams = get_ngram_counts(training_doc)
    # dirpath = "/Users/ada1024/Downloads/starter-code/gutenberg_data"
    # for dirpath, dirnames,filenames in os.walk(dirpath): 
    #     for file in filenames: 
    #         fp = os.path.join(dirpath,file)
    #         if "austen" not in fp: 
    #             do_experiment()
    #             add1_smoothing(doc,bigrams,trigrams)
    
   

    # word_frequency('/Users/ada1024/Downloads/starter-code/gutenberg_data')


    # 2.3 （1）
    # emma = "gutenberg_data/austen-emma.txt"
    # persuasion = "gutenberg_data/austen-persuasion.txt"
    # sense = "gutenberg_data/austen-sense.txt"
    # training_doc = make_doc([emma,persuasion,sense])
    # unigrams, bigrams, trigrams = get_ngram_counts(training_doc)
    # all_fp = []
    # curr = 0
    # dirpath = "/Users/ada1024/Downloads/starter-code/gutenberg_data"
    # for dirpath, dirnames,filenames in os.walk(dirpath):
    #     for file in filenames: 
    #         if "austen" not in file: 
    #             fp = os.path.join(dirpath,file)
    #             test_doc = make_doc([fp])
    #             add1_smoothing(test_doc,bigrams,trigrams)
    #             print(file,calc_text_perplexity(test_doc,bigrams,trigrams))

    # # # 2.3 （2）
    # # #austen
    # emma = "gutenberg_data/austen-emma.txt"
    # persuasion = "gutenberg_data/austen-persuasion.txt"
    # training_doc = make_doc([emma,persuasion])
    # unigrams, bigrams, trigrams = get_ngram_counts(training_doc)
    # sense = "gutenberg_data/austen-sense.txt"
    # test_doc = make_doc([sense])
    # add1_smoothing(test_doc,bigrams,trigrams)
    # print(calc_text_perplexity(test_doc,bigrams,trigrams))
    # #6.608959411840967


    # # # shakespeare
    # caesar = "gutenberg_data/shakespeare-caesar.txt"
    # hamlet = "gutenberg_data/shakespeare-hamlet.txt"
    # training_doc = make_doc([caesar,hamlet])
    # unigrams, bigrams, trigrams = get_ngram_counts(training_doc)
    # macbeth = "gutenberg_data/shakespeare-macbeth.txt"
    # test_doc = make_doc([macbeth])
    # add1_smoothing(test_doc,bigrams,trigrams)
    # print(calc_text_perplexity(test_doc,bigrams,trigrams))

    # # # chesterton
    # ball = "/Users/ada1024/Downloads/starter-code/gutenberg_data/chesterton-ball.txt"
    # brown = "/Users/ada1024/Downloads/starter-code/gutenberg_data/chesterton-brown.txt"
    # training_doc = make_doc([ball,brown])
    # unigrams, bigrams, trigrams = get_ngram_counts(training_doc)
    # thursday = "/Users/ada1024/Downloads/starter-code/gutenberg_data/chesterton-thursday.txt"
    # test_doc = make_doc([thursday])
    # add1_smoothing(test_doc,bigrams,trigrams)
    # print(calc_text_perplexity(test_doc,bigrams,trigrams))



    # all_fp = []
    # curr = 0
    # dirpath = "/Users/ada1024/Downloads/starter-code/gutenberg_data"
    # for dirpath, dirnames,filenames in os.walk(dirpath):
    #     for file in filenames: 
    #         if "austen" not in file: 
    #             fp = os.path.join(dirpath,file)
    #             test_doc = make_doc([fp])
    #             add1_smoothing(test_doc,bigrams,trigrams)
    #             print(file,calc_text_perplexity(test_doc,bigrams,trigrams))
    
    # fp = "/Users/ada1024/Downloads/starter-code/gutenberg_data/austen-emma.txt"
    # with open(fp,'r', encoding='latin1') as file:
    #     text =  file.read()
    #     doc = nlp(text)
    # one_doc_frequency(doc)

    # fp = "/Users/ada1024/Downloads/starter-code/gutenberg_data/austen-sense.txt"
    # with open(fp,'r', encoding='latin1') as file:
    #     text =  file.read()
    #     doc = nlp(text)
    # one_doc_frequency(doc)

    # fp = "/Users/ada1024/Downloads/starter-code/gutenberg_data/austen-persuasion.txt"
    # with open(fp,'r', encoding='latin1') as file:
    #     text = file.read()
    #     doc = nlp(text)
    # one_doc_frequency(doc)

    # emma = gutenberg.words("austen-emma.txt")
    # one_doc_frequency(emma)

    # sense = gutenberg.words("austen-sense.txt")
    # one_doc_frequency(sense)

    # p = gutenberg.words("austen-persuasion.txt")
    # one_doc_frequency(p)
    # dirpath = "/Users/ada1024/Downloads/starter-code/gutenberg_data"
    # c = read_all(dirpath)
    # for i,l in c.items(): 
    #     print(i,l)
    #     break
    
    pass
if __name__ == '__main__':
    # read_all("/Users/ada1024/Downloads/starter-code/gutenberg_data","txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", "-a",
                        type=argparse.FileType('rb'),
                        help="Content of articles")
    
    args = parser.parse_args()

    main(args)


# Discover NLP course materials authored by Julie Medero, Xanda Schofield, and Richard Wicentowski
# This work is licensed under a Creative Commons Attribution-ShareAlike 2.0 Generic License# https://creativecommons.org/licenses/by-sa/2.0/
