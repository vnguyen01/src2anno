import sys
import h5py
import numpy as np

sys.path.append("seq2seq-attn")

import preprocess

FILE_PATHS = {"anno" : "django/en-django/all.anno",
			  "code" : "django/en-django/all.anno"
			 }

def main():
	preprocess

if __name__ == "__main__":
	main()