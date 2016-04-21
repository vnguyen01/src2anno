import sys
import os

FILE_PATHS = {"train" : "data/en-django/django-train.hdf5",
			  "valid" : "data/en-django/django-val.hdf5",
			  "test" : "data/en-django/src-val.txt",
			  "dic_src" : "data/en-django/django.src.dict",
			  "dic_tar" : "data/en-django/django.tar.dict"}

def run_seq2seq_attn(ftrain, fvalid, fmodel, ftest, fpredict, dsrc, dtar):
	os.chdir('seq2seq_attn')

	#train
	os.system('th train.lua -data_file ' +\
	 ftrain + ' -val_data_file ' + \
	 fvalid + ' -savefile ' + fmodel)

	#predict
	os.system('th beam.lua -model ' + fmodel +\
	 ' -src_file ' + ftest + ' -output_file' + fpredict +\
	 ' -src_dict ' + dsrc + ' -targ_dict ' + dtar)


def main():
	ftrain = FILE_PATHS["train"]
	fvalid = FILE_PATHS["valid"]
	ftest = FILE_PATHS["test"]
	fpredict = "data/en-django/pred.txt"

	dsrc = FILE_PATHS["dic_src"]
	dtar = FILE_PATHS["dic_tar"]

	run_seq2seq_attn(ftrain, fvalid, fmodel, ftest, fpredict, dsrc, dtar)

if __name__ == "__main__":
	sys.exit(main())