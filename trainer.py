import nltk
import nltk.classify.util
import pickle
from nltk.classify import NaiveBayesClassifier
import requests
import unicodedata
import sys

def get_words(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


file_neg = open("negative.txt","r")
file_pos = open("positive.txt","r")

review_neg = []
review_pos = []

reviews_neg_unfiltered = file_neg.readlines()
reviews_pos_unfiltered = file_pos.readlines()


for i in range(len(reviews_neg_unfiltered)):
	if(i % 2 == 0):
		review_neg.append(reviews_neg_unfiltered[i].replace('.','').replace('\n','').replace(',','').replace('!','').replace('?',''))
		review_pos.append(reviews_pos_unfiltered[i].replace('.','').replace('\n','').replace(',','').replace('!','').replace('?',''))

pos_reviews_tagged = []
neg_reviews_tagged = []

for i in range(len(review_pos)):
	pos_reviews_tagged.append((review_pos[i], 'pos'))
	neg_reviews_tagged.append((review_neg[i],'neg'))

reviews_tagged = []



#FILTER OUT WORD WITH LENGTH < 3 AND CONVERT ALL WORDS INTO LOWER CASE
for (words, sentiment) in pos_reviews_tagged + neg_reviews_tagged:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    reviews_tagged.append((words_filtered, sentiment))
    
word_features = get_word_features(get_words(reviews_tagged))
training_set = nltk.classify.apply_features(extract_features, reviews_tagged)


#Training classifier 
classifier = nltk.NaiveBayesClassifier.train(training_set)

print "Saving the model for further use"
f = open('my_classifier.pickle','wb')
pickle.dump(classifier,f)
f.close()


