import sys
import h5py
import os
import numpy as np
from sklearn.cross_validation import train_test_split

FILE_PATHS = {"code" : "data/en-django/all.code",
			  "anno" : "data/en-django/all3.anno"
			 }

def split(data, target, seed, size):
	X_train, X_test, Y_train, Y_test = train_test_split(data, target, \
		random_state=seed, test_size=size)
	return X_train, X_test, Y_train, Y_test

def write2file(lst, fname):
	with open(fname, "w") as f:
		for line in lst:
			f.write(line + "\n")

def readin(fname):
	lst = []
	with open(fname, "r") as f:
		for line in f:
			lst.append(line.strip())
	return lst

def main():
	code = FILE_PATHS["code"]
	anno = FILE_PATHS["anno"]

	data = readin(code)
	target = readin(anno)

	X_train, X_test, Y_train, Y_test = split(data, target, 42, 0.2)

	write2file(X_train, "data/en-django/src-train.txt")
	write2file(X_test, "data/en-django/src-val.txt")
	write2file(Y_train, "data/en-django/targ-train.txt")
	write2file(Y_test, "data/en-django/targ-val.txt")

	os.chdir('seq2seq-attn')
	os.system('python preprocess.py --srcfile ../data/en-django/src-train.txt\
		--targetfile ../data/en-django/targ-train.txt --srcvalfile ../data/en-django/src-val.txt\
		--targetvalfile ../data/en-django/targ-val.txt --outputfile ../data/en-django/django')

if __name__ == "__main__":
	sys.exit(main())


