import sys
import os

def run_seq2seq_attn(ftrain, fvalid, fmodel, ftest, fpredict):
	os.chdir('seq2seq_attn')
	os.system('th train.lua -data_file ' +\
	 ftrain + ' -val_data_file ' + \
	 fvalid + ' -savefile ' + fmodel)
	os.system('th beam.lua -model ' + fmodel + ' -src_file ' + ftest + ' -out_file' + fpredict)

def main():
	run_seq2seq_attn(ftrain, fvalid, fmodel, ftest, fpredict)

if __name__ == "__main__":
	sys.exit(main())