import pandas as pd
import numpy as np
import json
import requests
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import xgboost
import string
from tkinter.filedialog import askopenfilename
import random

cwd = os.getcwd()

def generateModel():
    print("Please choose the file that contains training data.")
    dataFile = askopenfilename()
    train_data = pd.read_csv(dataFile)
    train_data.dropna(inplace=True, axis = 'rows', subset=['avgTopAcc'])
    clean_data = train_data.fillna(method='backfill', axis = 'columns')

    cols_to_use = []
    for i in range(1, 11):
        cols_to_use.append('pp{}'.format(i))
        cols_to_use.append('acc{}'.format(i))
    cols_to_use.append('avgStars')

    X_mixed = clean_data[cols_to_use].astype('float')
    y_mixed = clean_data['pp'].astype('float')
    X_train, X_valid, y_train, y_valid = train_test_split(X_mixed, y_mixed, test_size = 0.2)

    model = xgboost.XGBRegressor(n_estimators = 1500, learning_rate = 0.01, n_jobs = 6, early_stopping_rounds = 5)
    model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], verbose = False)

    predictions = model.predict(X_valid)
    stdv = np.sqrt(sklearn.metrics.mean_squared_error(predictions, y_valid))
    mean = np.mean(predictions - y_valid)
    print("Mean Error Squared = {}".format(sklearn.metrics.mean_squared_error(predictions, y_valid)))
    print("Mean Absolute Scaled Error = {}".format(sklearn.metrics.mean_absolute_percentage_error(predictions, y_valid)))

    return [model, stdv, mean]

def getLeaderboard(startRank, endRank, nation = "GLOBAL"):
    columns = ['rank', 'pp', 'playerName']
    for i in range(1, 11):
        columns.append("acc{}".format(i))
        columns.append("pp{}".format(i))
        columns.append("scoreRank{}".format(i))
    
    columns.append('avgTopAcc')
    columns.append('avgAcc')
    columns.append('avgStars')
    df = pd.DataFrame(columns = columns)
    currPage = np.floor(startRank / 50) + 1
    rank = startRank
    while(rank <= endRank):
        if(nation == "GLOBAL"):
            responseAPI = requests.get("https://scoresaber.com/api/players?page={}".format(currPage))
        else:
            responseAPI = requests.get("https://scoresaber.com/api/players?page={}&countries={}".format(currPage, nation))
        json_Data = responseAPI.json()

        for players in json_Data['players']:
            if(players['rank'] <  startRank):
                continue
            elif(rank > endRank):
                break
            rank += 1
            mapCount = 0
            playerScores = requests.get("https://scoresaber.com/api/player/{}/scores?limit=10&sort=top&page=1".format(players['id']))
            scores_json = playerScores.json()
            print("retrieving {} score".format(players['name']))
            row = []
            row.append(players['rank']) # target?
            row.append(players['pp']) # target?
            row.append("/{}".format(str(players['id'])))
            stars = 0
            topAcc = 0
            for scores in scores_json['playerScores']:
                if(scores['leaderboard']['ranked'] == False):
                    row.append(np.NaN)
                    row.append(np.NaN)
                    row.append(np.NaN)
                    continue
                mapCount += 1
                acc = scores['score']['modifiedScore'] / scores['leaderboard']['maxScore']
                row.append(acc)
                topAcc += acc
                row.append(scores['score']['pp'])
                row.append(scores['score']['rank'])
                stars += scores['leaderboard']['stars']

            row.append(topAcc / mapCount)
            row.append(players['scoreStats']['averageRankedAccuracy'] / 100)
            row.append(stars / mapCount)
            df = df.append(pd.Series(row, index=df.columns[:len(row)]), ignore_index=True)
        currPage += 1
    dataPath = cwd + "/data/{}".format(nation)
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
    df.to_csv(dataPath + "/{}EveryScoresFromRank{}Till{}.csv".format(nation, startRank, endRank), 
    index = True, columns = df.columns, float_format = '%.6f', na_rep = 'NULL', encoding = 'utf-8-sig')
    return df

def test(model, test_data, threshhold, nation):
    clean_test_data = test_data.fillna(method='backfill', axis = 'columns')
    cols_to_use = []
    for i in range(1, 11):
        cols_to_use.append('pp{}'.format(i))
        cols_to_use.append('acc{}'.format(i))
    cols_to_use.append('avgStars')
    X_test = clean_test_data[cols_to_use].astype('float')
    y_test = (clean_test_data['pp'].astype('float'))
    predictions = model.predict(X_test)
    result = pd.DataFrame(columns = ['name', 'pp', 'prediction', 'diff', 'sus'])
    result['name'] = test_data['playerName']
    result['pp'] = (clean_test_data['pp'].astype('float'))
    result['prediction'] = predictions
    result['diff'] = result['prediction'] - result['pp']
    result['sus'] = result.apply(lambda row : 1 if row['diff'] > threshhold else 0 , axis = 1)
    result = result.sort_values(by=['diff'], ascending = False)
    resultPath = cwd + "/data/results".format(nation)
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    result.to_csv(resultPath + "/{}results.csv".format(nation), index = True, columns = result.columns, float_format = '%.6f', na_rep = 'NULL')
    return result

modelStdv = generateModel()
model = modelStdv[0]
stdv = modelStdv[1]
mean = modelStdv[2]
print("Please enter start rank, end rank, and nation code you want to test a leaderboard against, separated by enter. If you want global leaderboard, enter GLOBAL.")
startRank = int(input())
endRank = int(input())
nationCode = input()
test_data = getLeaderboard(startRank, endRank, nationCode)
threshhold = mean + 2.576 * stdv
result = test(model, test_data, threshhold, nationCode)

