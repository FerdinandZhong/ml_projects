# coding=utf-8
# for chapter2
# recommend movie for people from the data from MovieLens

import math as math
from math import sqrt
import csv

def edclidean(prefs, person1, person2):
    both_list = [item for item in prefs[person1] if item in prefs[person2]]
    dis = math.sqrt(sum([(prefs[person1][item] - prefs[person2][item]) ** 2 for item in both_list]))
    return 1/dis


movie_list = {} #id:title
f = csv.reader(open("movies.csv", encoding = "utf-8"))
    # f.seek(0)
    # f_1 = pickle.load(f, encoding='bytes') 
next(f, None)
for row in f:
    (mid, title) = row[0:2]
    movie_list[mid] = title
    

pref_by_people = {} 
f = csv.reader(open('ratings.csv', encoding='utf-8'))
next(f, None)
for row in f:
    (uid, mid, rating) = row[0:3]
    if not uid in pref_by_people.keys():
    # 两个map嵌套
        pref_by_people[uid] = {}
    pref_by_people[uid][mid] = float(rating)

# 1 k-nn for users
def TransfromPref(pref):
#transfrom the struct of the pref
#{people:{movie:1}} -->> {movie:{people:1}}
    re_pref = {}
    for k1, v1 in pref.items():
        for k2, v2 in v1.items():
            if not k2 in re_pref.keys():
                re_pref[k2] = {}
            re_pref[k2][k1] = v2
    return re_pref

pref_by_movie = TransfromPref(pref_by_people)


def Pearson(pref = pref_by_movie, movie1 = '3', movie2 = '2'):
#calculating relevancy by Pearson corretation

    people_list = [person for person in pref[movie1].keys() if person in pref[movie2].keys()]
    n = len(people_list)
    if n == 0:
        return 0

    sum1 = sum([pref[movie1][person] for person in people_list])
    sum2 = sum([pref[movie2][person] for person in people_list])

    sumSq1 = sum([pref[movie1][person] ** 2 for person in people_list])
    sumSq2 = sum([pref[movie2][person] ** 2 for person in people_list])

    psum = sum([pref[movie1][person] * pref[movie2][person] for person in people_list])

    num = psum - sum1 * sum2 / n
    den = sqrt((sumSq1 - (sum1 ** 2) / n) * (sumSq2 - (sum2 ** 2) / n))

    if den == 0:
        return 0
    return num / den

def TopMatch(pref = pref_by_movie, movie = '3', n = 5):
# get the list that includes movies TopMatch the giving one
    scores = [(Pearson(pref_by_movie, movie, mov), mov) for mov in pref_by_movie.keys() if mov != movie]
    scores.sort(key = lambda x:x[0], reverse = True)
    return scores[0:n]


def CreateMatchList(pref = pref_by_movie):
# get the list of every movie's TopMatch
    match_list = {}
    for movie in pref.keys():
        match_list[movie] = TopMatch(pref, movie, 5)
    return match_list

match_list = CreateMatchList()
#print(len(match_list))

def get_recommended_items(pref = pref_by_people, match_list = match_list, user = '1'):
    try:
        user_ratings = pref[user]
    except KeyError:
        print("no user")
        return 0
    scores = {}
    totalsim = {}

    for movie, rating in user_ratings.items():
        for sim, sim_movie in match_list[movie]:
            if sim_movie in user_ratings.keys():
                continue
            if not sim_movie in scores.keys():
                scores[sim_movie] = sim * rating
                totalsim[sim_movie] = sim
            scores[sim_movie] += sim * rating
            totalsim[sim_movie] += sim

    rankings = [(scores[sim_movie]/totalsim[sim_movie], sim_movie, movie_list[sim_movie]) for sim_movie in scores.keys() if totalsim[sim_movie] != 0]

    rankings.sort(key=lambda x:x[0], reverse=True)
    return rankings[0:10]

print(get_recommended_items(pref = pref_by_people, match_list = match_list, user = '16'))

# phase 2 k-nn for users
def user_Pearson(person1, person2, pref = pref_by_people):
#calculating relevancy by Pearson corretation

    temp_movie_list = [movie for movie in pref[person1].keys() if movie in pref[person2].keys()]

    n = len(movie_list)
    if n == 0:
        return 0

    sum1 = sum([pref[person1][movie] for movie in temp_movie_list])
    sum2 = sum([pref[person2][movie] for movie in temp_movie_list])

    sumSq1 = sum([pref[person1][movie] ** 2 for movie in temp_movie_list])
    sumSq2 = sum([pref[person2][movie] ** 2 for movie in temp_movie_list])

    psum = sum([pref[person1][movie] * pref[person2][movie] for movie in temp_movie_list])

    num = psum - sum1 * sum2 / n
    den = sqrt((sumSq1 - (sum1 ** 2) / n) * (sumSq2 - (sum2 ** 2) / n))

    if den == 0:
        return 0
    return num / den


def UserTopMatch(person, pref = pref_by_people, n = 5):
# get the list that includes movies TopMatch the giving one
    scores = [(user_Pearson(person, person2, pref_by_people), person2) for person2 in pref_by_people.keys() if person2 != person]
    scores.sort(key = lambda x:x[0], reverse = True)
    return scores[0:n]


def CreateUserMatchList(pref = pref_by_people):
# get the list of every movie's TopMatch
    user_match_list = {}
    for person in pref.keys():
        user_match_list[person] = UserTopMatch(person, pref, 5)
    return user_match_list

user_match_list = CreateUserMatchList()

def get_user_recommended_items(pref = pref_by_people, match_list = user_match_list, target_user = '1'):
    try:
        sim_users = match_list[target_user]
        target_user_ratings = pref[target_user]
    except KeyError:
        print("no user")
        return 0
    scores = {}
    totalsim = {}

    for sim, user in sim_users:
        for movie, rating in pref[user].items():
            if movie in target_user_ratings.keys():
                continue
            if not movie in scores.keys():
                scores[movie] = sim * rating
                totalsim[movie] = sim
                continue
            scores[movie] += sim * rating
            totalsim[movie] += sim

    rankings = [(scores[movie]/totalsim[movie], movie, movie_list[movie]) for movie in scores.keys() if totalsim[movie] != 0]

    rankings.sort(key=lambda x:x[0], reverse=True)
    return rankings[0:10]

print(get_user_recommended_items(pref = pref_by_people, match_list = user_match_list, target_user = '16'))