import sys
import os

FILE_PATHS = {"train" : "data/en-django/django-train.hdf5",
			  "valid" : "data/en-django/django-val.hdf5",
			  "test" : "data/en-django/src-val.txt",
			  "model" : "django-model",
			  "dic_src" : "data/en-django/django.src.dict",
			  "dic_tar" : "data/en-django/django.tar.dict"}

def run_seq2seq_attn(ftrain, fvalid, fmodel, ftest, fpredict, dsrc, dtar):
	os.chdir('seq2seq-attn')
	#train
	os.system('th train.lua -data_file ' +\
	 "../" + ftrain + ' -val_data_file ' + \
	 "../" + fvalid + ' -savefile ' + fmodel +\
	  " -epochs 40 -num_layers 2 -word_vec_size 700 -rnn_size 700 -dropout 0.5")
	
	
	os.system('th beam.lua -model django-model_final.t7 -src_file ../data/en-django/src-val.txt -output_file pred.txt -src_dict ../data/en-django/django.src.dict -targ_dict ../data/en-django/django.targ.dict -max_sent_l 500')


def main():
	ftrain = FILE_PATHS["train"]
	fvalid = FILE_PATHS["valid"]
	fmodel = FILE_PATHS["model"]
	ftest = FILE_PATHS["test"]
	fpredict = "pred.txt"

	dsrc = FILE_PATHS["dic_src"]
	dtar = FILE_PATHS["dic_tar"]

	run_seq2seq_attn(ftrain, fvalid, fmodel, ftest, fpredict, dsrc, dtar)

if __name__ == "__main__":
	sys.exit(main())