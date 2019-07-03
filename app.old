from flask import Flask, render_template, request
import re
import sys
import requests
import pandas as pd
import numpy as np

# import preprocessing packages
import gensim
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import scipy
import nltk
nltk.download('punkt')
from gensim.corpora import Dictionary
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# import modeling packages
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import pickle

import random
from datetime import datetime

# open a file, where you stored the pickled data
w2v_file = open('data/best_w2v_model_alt.pkl', 'rb')
# dump information to that file
w2v_model = pickle.load(w2v_file)
w2v_file.close()

testing_df = pd.read_pickle('data/test_data_alt.pkl')

gloveFile = 'data/glove.6B.300d.txt'
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    with open(gloveFile, encoding="utf8" ) as f:
        content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

model = loadGloveModel(gloveFile)

def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = word_tokenize(letters_only_text.lower())

    # remove stopwords
    stop_words = stopwords.words("english")
    newStopWords = ['mryoucandoityourself', 'locksolid', 'petproof', 'tackstrip', 
                    'reflooring', 'drywallrepair', 'buildipedia', 'corefix', 'ghtl', 
                    'patreon', 'kraftmade', 'eztexture', 'renovision', 'yugipedia', 
                    'savogran', 'fibafuse', 'redbrand', 'how', 'to']
    stop_words.extend(newStopWords)
    stopword_set = set(stop_words)
    
    cleaned_words = set([w for w in words if w not in stopword_set])

    return cleaned_words

def compute_similarity(search, title):
    l1 =[];l2 =[]
    s1 = preprocess(search)
    t1 = preprocess(title)
    # form a set containing keywords of both strings  
    rvector = s1.union(t1)  
    for w in rvector: 
        if w in s1: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in t1: l2.append(1) 
        else: l2.append(0) 
    c = 0
    # cosine formula  
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cos_sim = c / float((sum(l1)*sum(l2))**0.5) 
    return cos_sim


#testing_df = pd.read_csv('data/test_data.csv', sep=',')

#### preprocess video data for word2vec classification
# function to remove punctuation and bad characters
def get_good_tokens(sentence):
	replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-z!?]+', '', token), sentence))
	removed_punctation = list(filter(lambda token: token, replaced_punctation))
	return removed_punctation

def w2v_preprocessing(df):
	"""
	All the preprocessing steps for word2vec are done in this function. All mutations are done on the dataframe itself. 
	So this function returns nothing.
    """
	df['title'] = df['title'].str.lower()
	df['document_sentences'] = df['title'].str.split('.')  # split texts into individual sentences
	df['tokenized_sentences'] = list(map(lambda sentences:
								list(map(nltk.word_tokenize, sentences)), df.document_sentences)) # tokenize sentences
	df['tokenized_sentences'] = list(map(lambda sentences: 
								list(map(get_good_tokens, sentences)), df.tokenized_sentences)) # remove unwanted characters
	df['tokenized_sentences'] = list(map(lambda sentences: 
								list(filter(lambda lst: lst, sentences)), df.tokenized_sentences)) # remove empty lists

def get_w2v_features(w2v_model, sentence_group):
    """ 
    Transform a sentence_group (containing multiple lists of words) into a feature vector. 
    It averages out all the word vectors of the sentence_group.
    """
    words = np.concatenate(sentence_group)  # words in text
    index2word_set = set(w2v_model.wv.vocab.keys())  # words known to model
    featureVec = np.zeros(w2v_model.vector_size, dtype="float32")
    # initialize a counter for number of words in a review
    nwords = 0
    # loop over each word in the comment and, if it is in the model's 
    # vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            featureVec = np.add(featureVec, w2v_model[word])
            nwords += 1.
    # divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def run_w2v_model(new_df):
	"""
	run new data against the trained word2vec model
	"""
	sentences = []
	for sentence_group in new_df.tokenized_sentences: sentences.extend(sentence_group)
	# set values for various parameters
	num_features = 200    # Word vector dimensionality
	min_word_count = 2    # Minimum word count
	num_workers = 4       # Number of threads to run in parallel
	context = 6           # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words

	# # initialize and train the model
	W2Vmodel = Word2Vec(sentences=sentences, sg=1, hs=0, workers=num_workers, size=num_features, 
						min_count=min_word_count, window=context, sample=downsampling, negative=5, iter=6)

	new_df['w2v_features'] = list(map(lambda sen_group: get_w2v_features(W2Vmodel, sen_group), new_df.tokenized_sentences))
	X_test_w2v = np.array(list(map(np.array, new_df.w2v_features)))

	label_encoder = LabelEncoder()
	label_encoder.fit(new_df.primary_category)
	new_df['category_id'] = label_encoder.transform(new_df.primary_category)
	prediction_proba = w2v_model.predict_proba(X_test_w2v)
	
	return prediction_proba

# function to find minimum and maximum position in list 
def max_pred(l):  
    # inbuilt function to find the position of maximum  
    maxpos = l.index(max(l))
    return maxpos

pd.options.display.max_columns=25


#Initialize app
app = Flask(__name__, static_url_path='/static')

#Standard home page. 'index.html' is the file in your templates that has the CSS and HTML for your app
@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template('index.html')


@app.route('/videos', methods=['GET', 'POST'])
def videos():
	punctuations = '''!()-[];:'",./?@#$%^&*_~'''
	error=None
	if request.method=='POST':
		sentence = request.form.get('statement')
		if not sentence.strip():
			print('No Sentence provided: double check')
			error = 'Whoops! One or more keywords (Deck, Carpet, Laminate, Drywall, Fence) was not entered, try again.'
			return render_template('index.html', error=error)
		else:
			testing_df['cos_sim'] = [compute_similarity(sentence, t) for t in testing_df.title]
			test_df = testing_df.sort_values('cos_sim', ascending=False).head(5)
			min_cos_sim = min(test_df.cos_sim)
			if min_cos_sim < 0.30:
				print('Not enough similar keywords')
				print(test_df.title,' -- ',test_df.cos_sim)
				error = 'Whoops! One or more keywords (Deck, Carpet, Laminate, Drywall, Fence) was not entered, try again.'
				return render_template('index.html', error=error)
			else:
				print(test_df.title,' -- ',test_df.cos_sim)

	X_test_w2v = np.array(list(map(np.array, test_df.w2v_features)))
	label_encoder = LabelEncoder()
	label_encoder.fit(test_df.primary_category)
	labels = [l for l in label_encoder.classes_]
	print(labels)
	#test_df['category_id'] = label_encoder.transform(test_df.primary_category)
	
	predictions_proba = w2v_model.predict_proba(X_test_w2v)

	titles=[]
	summaries=[]
	y_preds=[]
	p_cat=[]
	cat_id=[]
	l_idx=[]
	m_tools = []
	r_tools = []
	test_df['mentioned_tools'] = test_df.mentioned_tools.apply(lambda x: x[1:-1].split(', '))
	test_df['recommended_tools'] = test_df.recommended_tools.apply(lambda x: x[1:-1].split(', '))
	video_ids = [vId for vId in test_df.video_id]
	for i, vId in enumerate(video_ids):
		idx = test_df[test_df.video_id==vId].index
		titles.append(test_df.title[idx[0]].capitalize())
		summaries.append(test_df.summary[idx[0]])
		y_preds.append(round(max(list(predictions_proba[i]))*100,2))
		l_idx.append(list(predictions_proba[i]).index(max(predictions_proba[i])))
		p_cat.append(test_df.primary_category[idx[0]])
		cat_id.append(test_df.category_id[idx[0]])
		m_tools.append(test_df.mentioned_tools[idx[0]])
		r_tools.append(test_df.recommended_tools[idx[0]])
	
	print(m_tools)
	print(r_tools)

	print(l_idx)

	# deck_tools = ['circular saw','hand saw','chalk line','tape measure','measuring tape','level',
	# 			  'drill','square','plumb and square','pencil','post hole digger','shovel','clamps',
	# 			  'pry bar','nail puller','hammer','air compressor','pliers','wrench','jigsaw',
	# 			  'oscillating saw','reciprocating saw','utility knife','table saw','chop saw',
	# 			  'nail gun','belt sander','router']
	# carpet_tools = ['carpet knife','linoleum knife','duckbill napping shears',
	# 				'knee kicker','power stretcher','carpet row cutter','seam roller',
	# 				'seaming iron','stair tool','carpet tucker','stapler','electric staple gun',
	# 				'hammer staple','wall trimmer','heavy weight carpet roller','floor scraper',
	# 				'pencil','measuring tape','tape measure']
	# drywall_tools = ['Putty Knife', 'Drywall Saw', 'Drywall Screws', 
	# 				 'Utility Knife', 'Power Drill', 'Spackle', 
	# 				 'Tape Measure', 'Work Gloves', 'Pencil', 'Dust Mask'] 
	# laminate_tools = ['tapping block','pull bar','spacers','utility knife','hammer','pencil',
	# 				  'tape measure','measuring tape','square','router','drill','table saw',
	# 				  'miter saw','circular saw','hand saw','jigsaw','dividers','chalk line',
	# 				  'dead blow hammer','level','knee pads'] 
	# fence_tools = ['hammer','mason line','level','tape measure','measuring tape','chalk',
	# 			   'plumb bob','post hole digger','drill','deck screws','screwdriver',
	# 			   'sledge hammer','shovel','miter saw','circular saw','table saw','air compressor']

	#list_of_tools=[deck_tools, carpet_tools, drywall_tools, laminate_tools, fence_tools]

	return render_template('videos.html', video_ids=video_ids, titles=titles, m_tools=m_tools, r_tools=r_tools,
							y_preds=y_preds, labels=labels, l_idx=l_idx, summaries=summaries, p_cat=p_cat, cat_id=cat_id)


if __name__ == '__main__':
	#this runs your app locally
	app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
