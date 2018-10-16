import nltk
import sys
import os
import treetaggerwrapper
tagger = treetaggerwrapper.TreeTagger(TAGLANG='de')

path = os.getcwd()
os.chdir(path+"/"+sys.argv[1])

for novel in os.listdir(os.getcwd()):
	tokens = []
	for token in treetaggerwrapper.make_tags(tagger.tag_file(novel)):
		#print(token)	
		if token[1].startswith('N'):	
			tokens.append(token[2])
	novel_tokens = ([token for token in tokens
				if any(c.isalpha() for c in token) and token != "@card@"])
	with open(novel.split(".")[0]+"_lemma.txt", "w") as tok:
		for token in novel_tokens:
			tok.write(token+"\n")
