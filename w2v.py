from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import os
import random
import re
import json
from itertools import compress

import numpy as np
import string
#import tensorflow as tf
#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.metrics.pairwise import pairwise_distances
from keras.models import Sequential
from keras.layers.core import Dense, Activation
#from keras.losses import categorical_crossentropy




# Set random seeds
SEED = 2016
random.seed(SEED)
np.random.seed(SEED)


#################### Util functions #################### 

def build_dataset(words, vocabulary_size=50000):
	'''
	Build the dictionary and replace rare words with UNK token.
	
	Parameters
	----------
	words: list of tokens
	vocabulary_size: maximum number of top occurring tokens to produce, 
		rare tokens will be replaced by 'UNK'
	'''
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict() # {word: index}
	for word, _ in count:
		dictionary[word] = len(dictionary)
		data = list() # collect index
		unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0  # dictionary['UNK']
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count # list of tuples (word, count)
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary



data_index = 0

def generate_batch_cbow(data, batch_size, num_skips, skip_window,forbidden):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1 # [ skip_window target skip_window ]
	buffer = collections.deque(maxlen=span) # used for collecting data[data_index] in the sliding window
	# collect the first window of words
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	# move the sliding window
	i=0
	while(i<batch_size):
		mask = [1] * span
		mask[skip_window] = 0 
		batch[i, :] = list(compress(buffer, mask)) # all surrounding words
		labels[i, 0] = buffer[skip_window] # the word at the center
		if(labels[i,0] == forbidden):
			i=i-1
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
		i=i+1
	return batch, labels

def ohc1d (vsi,word):
	word=word.flatten()
	b=np.zeros((len(word),vsi))
	b[np.arange(len(word)),word] = 1
	return b

def ohc2d (vsi,word):
	resmat = np.zeros((word.shape[0],word.shape[1]*vsi))
	for i in range(word.shape[0]):
		resmat[i,]=ohc1d(vsi,word[i,].flatten()).flatten()
	return(resmat)

##This text must have 4 ###$ in between the document texts and punctuation free.
f=open("./maha.txt", "r")
contents =f.read()
f.close()
contents=contents.lower()
contents=contents.replace("."," ##$$ ##$$ ##$$ ##$$ ")
contents=contents.replace(","," ")
contents=contents.translate(string.punctuation)
#define parameters
epoch=10
windowsize=4
vsize=10000
words =contents.split()
print("Total string length : "+str(len(contents)))
print("Total number of words : "+str(len(words)))
nbatches=len(words)*epoch
print("Total number of batches to be trained for : "+str(nbatches))


data, count, dictionary, reverse_dictionary =  build_dataset(words, vocabulary_size=vsize)
forbidden=dictionary["##$$"]
print("Dictionary size : "+str(len(dictionary)))



##Define the NN
model = Sequential()
model.add(Dense(units=1024, input_dim=(windowsize)*vsize))
model.add(Activation('relu'))
model.add(Dense(units=vsize))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
#Train the NN
hj=0
while (hj< nbatches):
	batch, labels = generate_batch_cbow(data, 1024 ,windowsize, int(windowsize/2),forbidden)
	y_batch=ohc1d(vsize,labels)
	x_batch=ohc2d(vsize,batch)

	model.train_on_batch(x_batch, y_batch)
	print("batch no: "+str(hj)+" of "+str(nbatches))
	hj=hj+1024


