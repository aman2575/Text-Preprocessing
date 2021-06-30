# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:21:43 2021

@author: Aman
"""

import nltk

paragraph = '''    A rifle is a firearm designed to be fired from the shoulder, with a barrel that has a helical groove or pattern of grooves ("rifling") cut into the barrel walls. The raised areas of the rifling are called "lands," which make contact with the projectile (for small arms usage, called a bullet), imparting spin around an axis corresponding to the orientation of the weapon.

There are various types of rifles, most notably the Automatic rifle, the Bolt action rifle, the Lever-action rifle and the Semi-automatic rifle.
The table is sortable for every column. '''

#Cleaning the text
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
