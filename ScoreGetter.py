import string
import pandas as pd
import numpy as np
import json
import requests
import os
import random

def getPlayers(sampleNum, startPage, prob, nation = "GLOBAL"):
    columns = ['rank', 'pp', 'playerName']
    for i in range(1, 11):
        columns.append("acc{}".format(i))
        columns.append("pp{}".format(i))
        columns.append("scoreRank{}".format(i))
    
    columns.append('avgTopAcc')
    columns.append('avgAcc')
    columns.append('avgStars')
    df = pd.DataFrame(columns = columns)
    print(df.columns)
    count = 0
    page = startPage
    while(count < sampleNum):
        if(nation == "GLOBAL"):
            responseAPI = requests.get("https://scoresaber.com/api/players?page={}".format(page))
        else:
            responseAPI = requests.get("https://scoresaber.com/api/players?page={}&countries={}".format(page, nation))
        page += 1
        json_Data = responseAPI.json()
        for players in json_Data['players']:
            if(random.random() > prob): #expected number of last rank = 30000
                continue
            mapCount = 0
            count += 1
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
            if(count >= sampleNum):
                break
        print("{} and {}th player".format(page, count))
        
    cwd = os.getcwd()
    trainPath = cwd + "/data/train"
    if not os.path.exists(trainPath):
        os.makedirs(trainPath)
    try:
        if(nation == "GLOBAL"):
            df.to_csv(trainPath + "/ScoresTill{}Global.csv".format(page), index = True, columns = df.columns, float_format = '%.6f', na_rep = 'NULL')
        else:
            df.to_csv(trainPath + "/ScoresTill{}{}.csv".format(page, nation), index = True, columns = df.columns, float_format = '%.6f', na_rep = 'NULL')
    except PermissionError:
        print("Close the CSV file, and input OK")
        arg = input()
        if(arg == "OK"):
            if(nation == "GLOBAL"):
                df.to_csv(trainPath + "/ScoresTill{}Global.csv".format(page), index = True, columns = df.columns, float_format = '%.6f', na_rep = 'NULL')
            else:
                df.to_csv(trainPath + "/ScoresTill{}{}.csv".format(page, nation), index = True, columns = df.columns, float_format = '%.6f', na_rep = 'NULL')
    
print("Enter the number of samples that you want.")
requiredSample = int(input())
print("Enter the first page that you want your first data to start with.")
firstPage = int(input())
print("Enter the proportion of players that you want to have in your data.")
prob = float(input())
print("Please enter the country code. Press enter if you want a global data.")
nationCode = str(input())
if(nationCode == ""):
    getPlayers(requiredSample, firstPage, prob)
else:
    getPlayers(requiredSample, firstPage, prob, nationCode)
