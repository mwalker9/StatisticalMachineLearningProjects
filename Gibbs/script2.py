import pickle
import numpy as np
import random
import re
from numpy.random import dirichlet
import matplotlib.pyplot as plt
 
def randomly_assign_topics(docs_i, K):
	qs = []
	for arr in docs_i:
		docDist = []
		for word in arr:
			docDist.append(random.randint(0, K-1))
		qs.append(docDist)
	return qs
 
vocab = set()
docs = []
 
D = 472 # number of documents
K = 10 # number of topics
 
# open each file; convert everything to lowercase and strip non-letter symbols; split into words
for fileind in range( 1, D+1 ):
    foo = open( 'output%04d.txt' % fileind ).read()    
    tmp = re.sub( '[^a-z ]+', ' ', foo.lower() ).split()
    docs.append( tmp )
 
    for w in tmp:
        vocab.add( w )
 
# vocab now has unique words
# give each word in the vocab a unique id
ind = 0
vhash = {}
vindhash = {}
for i in list(vocab):
    vhash[i] = ind
    vindhash[ind] = i
    ind += 1
 
# size of our vocabulary
V = ind

# reprocess each document and re-represent it as a list of word ids
 
docs_i = []
for d in docs:
    dinds = []
    for w in d:
        dinds.append( vhash[w] )
    docs_i.append( dinds )
 
# ======================================================================
 
qs = randomly_assign_topics( docs_i, K )

c = np.zeros((D, V, K))

for doc, qList, i in zip(docs_i, qs, range(D)):
	for wordId, topicId in zip(doc, qList):
		c[i, wordId, topicId] += 1 

		
alphas = np.ones((K,1))[:,0]
gammas = np.ones((V,1))[:,0]

cvk = c.sum(axis=0)
cik = c.sum(axis=1)
ck = c.sum(axis=(0,1))
 
for iters in range(0,100):
    #p = compute_data_likelihood( docs_i, qs, topics, pdtm )
	p = 5
	print "Iter %d, p=%.2f" % (iters,p)
	for docDist, doc, i in zip(qs, docs_i, range(D)):
		Li = c.sum(axis=(1,2))
		Li = Li[i]
		for topicAssignment, word, idxInDoc in zip(docDist, doc, range(len(doc))):
			c[i, word, topicAssignment] -= 1
			cvk[word, topicAssignment] -= 1
			cik[i, topicAssignment] -= 1
			ck[topicAssignment] -= 1
			assert c[i, word, topicAssignment] >= 0
			probability = np.asarray([(cvk[word, topicNumber] + 1)/(ck[topicNumber] + V) * (cik[i, topicNumber] + 1)/(Li + K) for topicNumber in range(K)])
			probability = probability/probability.sum()
			newTopic = np.argmax(np.random.multinomial(1, probability)) 
			qs[i][idxInDoc] = newTopic
			c[i, word, newTopic] += 1
			cvk[word, newTopic] += 1
			cik[i, newTopic] += 1
			ck[newTopic] += 1

pickle.dump((qs, c), open("qs.pickle", "w"))