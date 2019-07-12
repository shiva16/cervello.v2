from flask import Flask, render_template, request
import nltk 
import numpy as np
import spacy
import en_coref_md
from nltk import word_tokenize,sent_tokenize
from gensim.models import Word2Vec
import xml.etree.ElementTree as ET
import pymongo
from pymongo import MongoClient
import pprint
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.corpus import stopwords
import csv
import xml.etree.ElementTree as ET





client = MongoClient()
db = client.project
collection = db.pjcollection
nltk.download('punkt') 
app = Flask(__name__)


@app.route('/')
def student():
   return render_template('student.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
	if request.method == 'POST':
		result = request.form
		x=result['X']
		y=result['Y']




		tree = ET.parse("abbreviation.xml")
		root = tree.getroot()
		
		for app in root.findall('parts'):
			for l in app.findall('part'): 
				if (l.attrib['name'])==x:
						abbx = l.attrib['abbreviation']
						print(abbx)
				else: 
					abbx = 0
				if (l.attrib['name'])==y:
						abby = l.attrib['abbreviation']
						print(abby)
				else:
					abby = 0
		with open('bams.csv','rt')as f:
			data = csv.reader(f)
			for row in data:
				if (row[0]==str(abbx) and row[1]==str(abby)):
					print("hi")
					score = row[2]
					print(f"BAMS score of the TERMS:",row[2])            
		
		collection.create_index([('entry.summary', 'text')])
		list_x = collection.find( { "$text": { "$search": x } } )
		list_y = collection.find( { "$text": { "$search": y } } )
		#list_final = dict(list_x.items() & list_y.items())
		"""for item in list_x.keys( ):
			if list_y.has_key(item):
				list_final.append(item)"""

		print(list_x.count())
		print(list_y.count())
		if list_x.count()>list_y.count():
			list_final = [value for value in list_y if value in list_x]

		else:
			list_final = [value for value in list_x if value in list_y]	

		
		print(list_final)
			
			           	
		nlp = en_coref_md.load()
		# split the the text in the articles into sentences
		sentences =[]
		for elem in list_final:
			doc=nlp(elem['entry']['summary'])
			sentences.append(sent_tokenize(elem['entry']['summary']))  

		# flatten the list
		sentences = [y for x in sentences for y in x]
		# remove punctuations, numbers and special characters
		clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
		# make alphabets lowercase
		clean_sentences = [s.lower() for s in clean_sentences]
		doc._.has_coref
		doc._.coref_clusters
		doc._.coref_resolved
		nltk.download('stopwords')		
		stop_words = stopwords.words('english')
		# function to remove stopwords
		def remove_stopwords(sen):
			sen_new = " ".join([i for i in sen if i not in stop_words])
			return sen_new
		# remove stopwords from the sentences
		clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

		word_embeddings = {}
		f = open('glove.6B.100d.txt', encoding='utf-8')
		for line in f:
		    values = line.split()
		    word = values[0]
		    coefs = np.asarray(values[1:], dtype='float32')
		    word_embeddings[word] = coefs
		f.close()

		sentence_vectors = []
		for i in clean_sentences:
			if len(i) != 0:
				v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
			else:
				v = np.zeros((100,))
			sentence_vectors.append(v)

		len(sentence_vectors)


		sim_mat = np.zeros([len(sentences), len(sentences)])
		for i in range(len(sentences)):
			for j in range(len(sentences)):
				if i != j:
					sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

		nx_graph = nx.from_numpy_array(sim_mat)
		scores = nx.pagerank(nx_graph)
		ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)		
		sn = 1
		for i in range(sn):
			print(ranked_sentences[i][1])


		model = Word2Vec.load("word2vec.model")
		sim = model.similarity(x,y)
		print(sim)
		print(result)



				
	return render_template("final.html",sim=sim,list_final=list_final,ranked_sentences=ranked_sentences[0][1],x=x,y=y,score=score)

if __name__ == '__main__':
   app.run(debug = True)


