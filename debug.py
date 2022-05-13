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

#for i in range(len(reviews_pos_unfiltered)):#
#	if(i % 2 == 0):
#		review_pos.append(reviews_pos_unfiltered[i].replace('.','').replace('\n',''))

for i in range(len(reviews_neg_unfiltered)):
	if(i % 2 == 0):
		review_neg.append(reviews_neg_unfiltered[i].replace('.','').replace('\n',''))
		review_pos.append(reviews_pos_unfiltered[i].replace('.','').replace('\n',''))

pos_reviews_tagged = []
neg_reviews_tagged = []

for i in range(len(review_pos)):
	pos_reviews_tagged.append((review_pos[i], 'pos'))



for i in range(len(review_neg)):
	neg_reviews_tagged.append((review_neg[i],'neg'))

reviews_tagged = []



#FILTER OUT WORD WITH LENGTH < 3 AND CONVERT ALL WORDS INTO LOWER CASE
for (words, sentiment) in pos_reviews_tagged + neg_reviews_tagged:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    reviews_tagged.append((words_filtered, sentiment))
    
word_features = get_word_features(get_words(reviews_tagged))


locationUrlFromLatLong = "https://developers.zomato.com/api/v2.1/reviews?res_id=%d&count=5" % int(sys.argv[1])
header = {"User-agent": "curl/7.43.0", "Accept": "application/json", "user_key": "ac78c6cc1c848720eb6adb47002113e4"}

response = requests.get(locationUrlFromLatLong, headers=header)
data_dict=response.json()
user_reviews_list=data_dict['user_reviews']

reviewList=[]

for i in range(0,5,1):
    user_review_dict=user_reviews_list[i]
    user_review_dict=user_review_dict['review']
    review=user_review_dict['review_text']
    review=unicodedata.normalize('NFKD', review).encode('ascii', 'ignore')
    reviewList.append(review)

##Load Classifier
f = open('my_classifier.pickle','rb')
classifier = pickle.load(f)
f.close()

#classifier.show_most_informative_features()


#Get Probability Classification
length_review = len(reviewList)
final_rating = 0

for i in range(length_review):
	prob_class =  classifier.prob_classify(extract_features(reviewList[i].split()))
	temp = classes_prob = prob_class.samples()
	high_prob = prob_class.prob(classes_prob[0])
	low_prob = prob_class.prob(classes_prob[1])
	if temp[0] == 'pos':
		final_rating  = final_rating + high_prob
	else:
		final_rating  = final_rating + low_prob
	
#Save Classifier to file my_classifier.pickle

print "Final Rating Calculated = %.2f out of 5" % final_rating





