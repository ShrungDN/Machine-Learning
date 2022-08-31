from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

def process_message(message, lower_case=True, stem=True, stop_words=True):
  if lower_case:
    message=message.lower()
  
  message = re.sub('[\.\?\!{}\"\'\\\\\/\,\+\-\;\*]',' ', message)
  message = re.sub('subject: ', '', message)
  message = re.sub('\:$\%\#\[\]\<\>', '', message)
  message = re.sub(' [0-9]* ', '', message)

  k = 4
  words = word_tokenize(message)
  words = [w for w in words if len(w) >= k]  # here k is a hyperparameter

  if stop_words:
    sw = stopwords.words('english')
    words = [w for w in words if w not in sw]

  if stem:
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
  return words


class NB():
  def __init__(self, train_data):
    self.mails = train_data['message'].copy()
    self.labels = train_data['label'].copy()
    self.train()
  
  def train(self):
    self.create_dictionary()
    self.d = len(self.dictionary) 
    self.n = len(self.labels)

    self.p = self.n1/self.n

    self.p1j = {}
    self.p0j = {}
  
    for word in self.dictionary.keys():
      j = self.dictionary[word]

      self.p1j[j] = (self.spam.get(word,0) + 1) /(self.n1 + len(self.spam.keys()))
      self.p0j[j] = (self.ham.get(word,0) + 1)/(self.n0 + len(self.ham.keys()))

  def create_dictionary(self):
    self.dictionary = {}
    self.n1 = 0
    self.n0 = 0
    self.spam = {}
    self.ham = {}
    word_count = 0
    for (idx, line) in enumerate(self.mails):
      processed_line = process_message(line)
      if self.labels[idx] == 0:
        self.n0 += 1
      else:
        self.n1 += 1
      
      words_appeared = []

      for word in processed_line:

        if word not in self.dictionary:
          self.dictionary[word] = word_count
          word_count += 1

        if word not in words_appeared:
          if self.labels[idx] == 0:
            if word not in self.ham:
              self.ham[word] = 1
            else:
              self.ham[word] += 1

          if self.labels[idx] == 1:
            if word not in self.spam:
              self.spam[word] = 1
            else:
              self.spam[word] += 1
          
          words_appeared.append(word)

  def classify(self, processed_message):
    p1 = 1
    p0 = 1

    for word in self.dictionary.keys():
      if word in processed_message:
        p1 *= self.p1j.get(self.dictionary.get(word))
        p0 *= self.p0j.get(self.dictionary.get(word))
      else:
        p1 *= (1 - self.p1j.get(self.dictionary.get(word)))
        p0 *= (1 - self.p0j.get(self.dictionary.get(word)))
    
    if p1 > p0:
      return 1
    else:
      return 0

  def predict(self, test_data):
    result = []
    for message in test_data: 
      processed_message = process_message(message) 
      result.append(int(self.classify(processed_message)))
    return result