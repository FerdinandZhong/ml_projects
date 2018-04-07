# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:29:33 2017

@author: Ferdinand
"""

import numpy as np
import csv
import pandas as pd

header = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('D:\\movielens project\\ml-latest-small\\ml-latest-small\\ratings.csv', sep = ',', names = header)
df = df.iloc[1:]

movie_list = {} #id:title
f = csv.reader(open("D:\\movielens project\\ml-latest-small\\ml-latest-small\\movies.csv", encoding = "utf-8"))
next(f, None)
for row in f:
    (mid, title) = row[0:2]
    movie_list[mid] = title

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

movie_ids = df.item_id.unique()
movies_index = {}

for index in range(len(movie_ids)):
    movies_index[movie_ids[index]] = index

from sklearn import cross_validation as cv

train_data,test_data = cv.train_test_split(df, test_size = 0.25)
train_data_matrix = np.zeros((n_users,n_items))

for line in train_data.itertuples():
    train_data_matrix[int(line[1])-1, movies_index[line[2]]] = line[3]

test_data_matrix = np.zeros((n_users, n_items))

for line in test_data.itertuples():
    test_data_matrix[int(line[1])-1, movies_index[line[2]]] = line[3]

from sklearn.metrics.pairwise import pairwise_distances

user_similarity = pairwise_distances(train_data_matrix, metric = "cosine")

def predict(ratings, similarity):
     mean_user_rating = ratings.mean(axis=1)
     #You use np.newaxis so that mean_user_rating has same format as ratings
     ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
     pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
     return pred

user_prediction = predict(train_data_matrix, user_similarity)

import operator
def give_result(user_prediction = user_prediction, movie_ids = movie_ids, userId = '1'):
    result = user_prediction[int(userId)-1,]
    scores = {}
    for index in range(len(result)):
        scores[movie_ids[index]] = result[index]
    sorted_score = sorted(scores.items(), key=operator.itemgetter(1), reverse = True)    
    final_result = [(row[0], movie_list[row[0]]) for row in sorted_score[0:5]]
    return final_result;
        

give_result(userId = '16')

# evaluation
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print( str(rmse(user_prediction, test_data_matrix)))