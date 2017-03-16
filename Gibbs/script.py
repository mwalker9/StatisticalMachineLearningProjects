import pickle
import numpy as np
import random
import re
from numpy.random import dirichlet
import matplotlib.pyplot as plt

def compute_data_likelihood(docs_i, qs, topics, pdtm):
	ll = 0
	for doc, i in zip(docs_i, range(len(docs_i))):
		for j in range(len(doc)):
			ll += np.log(topics[doc[j], qs[i][j]])
	return ll
		
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
 
# topic distributions
topics = np.zeros((V,K))
for k in range(K):
	topics[:, k] = dirichlet(gammas)
# how should this be initialized? Random--sums to 1 when marginalize out V for each K--probability word V is in topic K

# per-document-topic distributions
pdtm = np.zeros((K,D))
for i in range(D):
	pdtm[:, i] = dirichlet(alphas)
# how should this be initialized? Random--sums to 1 when marginalize out K for each D--probability topic K is in document D
likelihood = []
for iters in range(0,100):
	p = compute_data_likelihood( docs_i, qs, topics, pdtm )
	print "Iter %d, p=%.2f" % (iters,p)
	likelihood.append(p)
    # resample per-word topic assignments qs
	for docDist, doc, i in zip(qs, docs_i, range(D)):
		for topicAssignment, word, idxInDoc in zip(docDist, doc, range(len(doc))):
			c[i, word, topicAssignment] -= 1
			assert c[i, word, topicAssignment] >= 0
			probability = np.zeros(K)
			for topicNumber in range(K):
				probability[topicNumber] = pdtm[topicNumber, i] * topics[word, topicNumber]
			probability = probability/probability.sum()
			qs[i][idxInDoc] = np.argmax(np.random.multinomial(1, probability))
			c[i, word, qs[i][idxInDoc]] += 1
	
    # resample per-document topic mixtures pdtm
	cik = c.sum(axis=1)
	for i in range(D):
		pdtm[:, i] = dirichlet(alphas + np.asarray([cik[i, k] for k in range(K)]))
	
			
    #resample topics
	cvk = c.sum(axis=0)
	for k in range(K):
		topics[:, k] = dirichlet(gammas + np.asarray([cvk[v, k] for v in range(V)]))
		
pickle.dump((qs, topics, pdtm, likelihood), open("topicResults.pickle", "w"))