from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import os
from collections import Counter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm

def make_Dictionary(root_dir):
   all_words = []
   emails = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
   for mail in emails:
        with open(mail) as m:
            for line in m:
                words = line.split()
                all_words += words
   dictionary = Counter(all_words)
   # if you have python version 3.x use commented version.
   list_to_remove = list(dictionary)
   #list_to_remove = dictionary.keys()
   for item in list_to_remove:
       # remove if numerical. 
       if item.isalpha() == False:
            del dictionary[item]
       elif len(item) == 1:
            del dictionary[item]
    # consider only most 3000 common words in dictionary.
   dictionary = dictionary.most_common(3000)
   return dictionary

def extract_features(mail_dir):
  files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
  features_matrix = np.zeros((len(files),3000))
  train_labels = np.zeros(len(files))
  count = 0;
  docID = 0;
  for fil in files:
    with open(fil) as fi:
      for i,line in enumerate(fi):
        if i == 2:
          words = line.split()
          for word in words:
            wordID = 0
            for i,d in enumerate(dictionary):
              if d[0] == word:
                wordID = i
                features_matrix[docID,wordID] = words.count(word)
      train_labels[docID] = 0;
      filepathTokens = fil.split('\\')
      lastToken = filepathTokens[len(filepathTokens) - 1]
      if lastToken.startswith("spmsg"):
          train_labels[docID] = 1;
          count = count + 1
      docID = docID + 1
  return features_matrix, train_labels

TRAIN_DIR = "./train-mails"
TEST_DIR = "./test-mails"
dictionary = make_Dictionary(TRAIN_DIR)
# using functions mentioned above.
features_matrix, labels = extract_features(TRAIN_DIR)
test_feature_matrix, test_labels = extract_features(TEST_DIR)

ga = GaussianNB()
mu = MultinomialNB()
be = BernoulliNB()

ga.fit(features_matrix, labels)
mu.fit(features_matrix, labels)
be.fit(features_matrix, labels)

predict_ga = ga.predict(test_feature_matrix)
predict_mu = mu.predict(test_feature_matrix)
predict_be = be.predict(test_feature_matrix)

ga_ = cm(test_labels, predict_ga)
mu_ = cm(test_labels, predict_mu)
be_ = cm(test_labels, predict_be)


sns.heatmap(ga_, annot = True)
plt.imsave('GaussianNB.png')

sns.heatmap(mu_, annot = True)
plt.imsave('MultinomialNB.png')

sns.heatmap(be_, annot = True)
plt.imsave('BernoulliNB.png')

