#!/usr/bin/python
import sys

# This script extracts n-grams from a set of PREPROCESSED text files (train, set, valid)

data_dir = sys.argv[1]
ngram_order = int(sys.argv[2])

train_file = data_dir + "/train.txt"

# Create input and output files for train data
fin = open(train_file)

gram = str(ngram_order) + "grams"
fout = open(train_file.replace("txt", gram), 'w')

# Sweeping through the train file
for line in fin.readlines():

	sentence = line.strip().split()

	# adding start tokens
	sentence = ["<s>"] + sentence

	for i, word in enumerate(sentence):

		ngram = list()
		if i == 0:
			continue

		# if i == len(sentence)-1:
		# 	continue

		for j in range(max(0,i+1-ngram_order), i+1):

			if len(ngram) < ngram_order:
				ngram.append(sentence[j])
		
		while len(ngram) < ngram_order:
			ngram = ["<s>"] + ngram


		ngram_text = " ".join(ngram)

		# print ngram_text
		fout.write(ngram_text + "\n")


fin.close()
fout.close()

valid_file = data_dir + "/valid.txt"
fin = open(valid_file)
fout = open(valid_file.replace("txt", gram), 'w')

# Sweeping through the train file
for line in fin.readlines():

	sentence = line.strip().split()

	# adding start tokens
	sentence = ["<s>"] + sentence

	for i, word in enumerate(sentence):

		ngram = list()
		if i == 0:
			continue

		# if i == len(sentence)-1:
		# 	continue

		for j in range(max(0,i+1-ngram_order), i+1):

			if len(ngram) < ngram_order:
				ngram.append(sentence[j])
		
		while len(ngram) < ngram_order:
			ngram = ["<s>"] + ngram


		ngram_text = " ".join(ngram)

		# print ngram_text
		fout.write(ngram_text + "\n")


fin.close()
fout.close()

valid_file = data_dir + "/test.txt"
fin = open(valid_file)
fout = open(valid_file.replace("txt", gram), 'w')

# Sweeping through the train file
for line in fin.readlines():

	sentence = line.strip().split()

	# adding start tokens
	sentence = ["<s>"] + sentence

	for i, word in enumerate(sentence):

		ngram = list()
		if i == 0:
			continue

		# if i == len(sentence)-1:
		# 	continue

		for j in range(max(0,i+1-ngram_order), i+1):

			if len(ngram) < ngram_order:
				ngram.append(sentence[j])
		
		while len(ngram) < ngram_order:
			ngram = ["<s>"] + ngram


		ngram_text = " ".join(ngram)

		# print ngram_text
		fout.write(ngram_text + "\n")


fin.close()
fout.close()
