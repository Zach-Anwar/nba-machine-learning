import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

current_year = 2025 # ending year of the upcoming NBA season
maxOut = 2 
# limits the max salary out a team can have in a trade recomendation. A value of 2 makes the max salary of a trade the sum of the 
#two largest salaries on the team


team_path = '23team.csv'
players_path = '23players.csv'
team_data_path = 'testData.csv'

def create_sparse_matrix(player_training_data, team_training_data, minWinshare, age_reg, perf_reg): 

    # creates sparse matrix Y for collabrative filtering training

    players = player_training_data['PLAYER'].unique().tolist() 
    team_df_length = len(team_training_data)
    players_length = len(players)
    players.sort() #sort players alphabetically
    Y = np.zeros((team_df_length, players_length)) 
    i = 0

    print("Creating sparse matrix Y")

    for index1, team in team_training_data.iterrows(): #for each team row in the df

        print(str(index1) + "/" + str(team_df_length - 1))

        team_players = player_training_data[(player_training_data["TEAM"] == team["ABR"]) & (player_training_data["Year"] == team["Year"])]
        # gets all the players on that team

        for index, player in team_players.iterrows(): 
            #adds the winshare values for each team player to the relevant row in matrix Y

            name = player["PLAYER"]
            relative_performance = find_age_adj(player["AGE"], age_reg, perf_reg)
            year = player["Year"]
            current_performance = find_age_adj(current_year - year + player["AGE"], age_reg, perf_reg)
            adjust_for_age = relative_performance/current_performance 
            # uses the results of polynomial regression to adjust the winshare for the player's age at the time

            j = players.index(name)
            Y[i,j] = ((player["Winshare"] + minWinshare) * adjust_for_age )- minWinshare

        i += 1
    return Y

def loss_function(Y, alpha, W, Z):
    # calculates the loss function value for a random player value
    # this is used for an iteration of the stochastic gradient descent method and updates the W and Z matrices

    team_n = random.randint(0, Y.shape[0] - 1)
    players = Y[team_n]
    valid_indices = np.where(players != 0)[0]
    num_valid = len(valid_indices)
    player_choice = random.randint(0, num_valid - 1)
    player_n = valid_indices[player_choice] # chooses a random player team value

    prediction = np.dot(W[player_n], Z[team_n]) # calculates the current prediction of the model

    error = prediction - Y[team_n, player_n] # error of prediction is found

    W[player_n] -= alpha * error * Z[team_n]
    Z[team_n] -= alpha * error * W[player_n] # W and Z are updated in the relevant rows accoridngly

    return W,Z, error

def find_age_adj(age, age_reg, performance_reg): 

    #finds the closest value to an input age that has a winshare value in the polynomial regression results

    closest_ind = -1
    closest = 100
    for i in range(age_reg.shape[0]):
        if closest > abs(age - age_reg[i][0]):
            closest = abs(age - age_reg[i][0])
            closest_ind = i
        else:
            return performance_reg[closest_ind]
    return performance_reg[closest_ind]

def train_collab(accurate_preds, alpha, W, Z, player_training_data, team_training_data, min_winshare, age_reg, perf_reg):

    #applies stochastic gradient descent on W and Z so the collabrative filtering model can learn how to calculate winshare

    players = player_training_data['PLAYER'].unique() 
    Y = create_sparse_matrix(player_training_data, team_training_data, min_winshare, age_reg, perf_reg)
    error_check = 0

    while (error_check < accurate_preds): # model has to get 100 predictions correct in a row to end training

        W, Z, error = loss_function(Y, alpha, W, Z)
        #iteration of stochastic gradient descent

        if (error < 0.01): #checks error is within and acceptable range

            error_check += 1

            if (error_check > 50):
                print(str(error_check) + " predictions correct in a row")
        else:
            error_check = 0

    predictions = np.dot(Z, W.T)
    np.save('array_data.npy', predictions)
    return predictions

def get_item_matrix(training_data): #returns feature matrix of the players 

    players = training_data[["Pts", "AST", "FG%", "OREB%", "TOV", "FT%", "FGA", "OREB", "3P%", "3PM", "BLK", "MIN", "DEFRTG", "DREB", "DREB%"]].values
    return (players - players.min()) / (players.max() - players.min()) # normalises data


def get_user_matrix(training_data): #returns feature matrix of the teams 
    training_data["MIN"] = training_data["MIN"]/training_data["GP"] # converts minute stat to per game instead of total
    teams = training_data[["PTS", "AST", "FG%", "OREB%", "TOV", "FT%", "FGA", "OREB", "3P%", "3PM", "BLK", "MIN", "DefRtg", "DREB", "DREB%"]].values
    return (teams - teams.min()) / (teams.max() - teams.min()) # normalises data
    

def winshare(season, year):
    
    #Calculate the total Win Shares (winshare) by combining Offensive Win Shares (OWS) and Defensive Win Shares (DWS).

    offense = ows(season, year)
    defense = dws(season, year)
    if offense is not None and defense is not None:
        winshare = offense + defense
        return winshare
    else:
        print("Error: Unable to calculate Win Shares. Check the input values.")
        return None


def calculate_lppp(year): 
    
    #calculates the average league points per possession for given year of a season

    possessions = 0
    points = 0
    team_rows = team_rows = team_df[team_df['Year'] == year]
    for index, team in team_rows.iterrows():
        possessions +=  float(team["POS"].replace(',', ''))
        points += float(team["PTS"]) * 82
    return points/possessions


def calculate_lppg(year):

    #calculates the average league points per game for given year of a season

    points = 0
    team_rows = team_rows = team_df[team_df['Year'] == year]
    for index, team in team_rows.iterrows():
        points += float(team["PTS"])
    return points/30

def calculate_lp(year):
    
    #calculates the average league pace given year of a season

    pace = 0
    team_rows = team_rows = team_df[team_df['Year'] == year]
    for index, team in team_rows.iterrows():
        pace += team["PACE"] 
    return pace/30

def ows(season, year):

    # calculates offensive winshares
    #Parameters:
    #-- pp: Points Produced (PP) \/
    #- lppp: League's average points per possession (LPPP) \/
    #-- op: Player's offensive possessions (OP) \/
    #- lppg: League's average points per game (LPPG) \/
    #--- tp: Team's pace possesions per 48 misn (TP) \/
    #- lp: league pace (LP) \/

    lppp = calculate_lppp(year)
    lppg = calculate_lppg(year)
    team = str(season["TEAM"])
    team_rows = team_df[(team_df['ABR'] == team) & (team_df['Year'] == year)]
    Team_MP = float(team_rows["GP"])
    tp = float(team_rows["PACE"])
    lp = calculate_lp(year)
    gp = float(season["GP"])

    op = offensive_poss(Team_MP, float(season["FGM"]), float(season["Pts"]), float(season["FTM"]), float(season["FGA"]), float(season["GP"]), float(team_rows["PTS"]),float(team_rows["AST"]), float(season["AST"]), float(team_rows["FGM"]), float(team_rows["FTM"]), float(team_rows["FGA"]), float(team_rows["TOV"]), float(season["FTA"]), float(team_rows["FTA"]), float(team_rows["OREB"]), float(season["OREB"]), float(team_rows["OREB%"])/100)

    pp = calculate_pp(float(season["FGA"]),float(season["FTA"]),float(season["TOV"]),float(season["OFFRTG"]), op, float(season["GP"]))

    pp2 = calculate_basketball_production(Team_MP, gp * float(season["FGM"]), gp * float(season["3PM"]), gp * float(season["Pts"]), gp * float(season["FTM"]), gp * float(season["FGA"]), gp * float(season["AST"]), Team_MP*float(team_rows["FGM"]), Team_MP*float(team_rows["3PM"]), Team_MP*float(team_rows["PTS"]), Team_MP*float(team_rows["FTM"]), Team_MP*float(team_rows["FTA"]), Team_MP*float(team_rows["FGA"]), gp * float(season["OREB"]), Team_MP*float(team_rows["OREB"]), Team_MP*float(team_rows["AST"]), gp, float(team_rows["OREB%"])/100, Team_MP*float(team_rows["TOV"]))

    numerator = pp - 0.92 * (lppp * op) # calculates marginal offense

    denominator = 0.32 * (lppg) * (tp / lp) # calculates marginal points per game

    if denominator != 0:
        ows = ((numerator / denominator)/Team_MP) * 82
        return ows
    else:
        print("Error: Division by zero. Check the input values.")
        return None
    

def dws(season, year):
    
    # -- pmp: Player's minutes played (PMP) \/
    # --- tmp: Team's total minutes played (TMP)\/
    # --- tdp: Team's total defensive possessions (TDP) \/
    # -- dr: Player's defensive ratinga (DR) \/

    pmp = float(season["MIN"]) * float(season["GP"])
    team = str(season["TEAM"])
    team_rows = team_df[(team_df['ABR'] == team) & (team_df['Year'] == year)]
    Team_MP = float(team_rows["GP"])
    tmp = float(team_rows["MIN"])
    tdp = Defensive_pos(float(team_rows["OpFGA"]), float(team_rows["OpTOV"]), float(team_rows["OpFTA"]), float(team_rows["OpOREB"])) * Team_MP
    lppp = calculate_lppp(year)
    lppg = calculate_lppg(year)
    dr = float(season["DEFRTG"])
    tp = float(team_rows["PACE"])
    lp = calculate_lp(year)

    numerator = (pmp / (tmp*5)) * tdp * ((1.08 * lppp) - (dr / 100)) # calculates marginal defense

    denominator = 0.32 * lppg * (tp / lp) # calculates marginal points per game

    if denominator != 0:
        dws = ((numerator / denominator)/Team_MP) * 82
        return dws
    else:
        print("Error: Division by zero. Check the input values.")
        return None

    

def approx_value(points, rebounds, assists, steals, blocks, field_goals_missed, free_throws_missed, turnovers): 

    # calculates the approximate value of a player using their season stats
    # this is used to estimate a player's trade value

    credits = (points) + (rebounds) + (assists) + (steals) + (blocks) - (field_goals_missed) - (free_throws_missed) - (turnovers)
    av = (credits * 3/4) * 2
    return av


def predict_trade(playersIn, playersOut, team):

    # calucates the value in winshares of a trade by summing the prediction of players in minus players out

    prediction = 0
    for player in playersIn:
        prediction += predict(player, team)
    for player in playersOut:
        prediction -= predict(player, team)

    return prediction

def max_knapsack(players, maxSalary, maxTradeValue):

    # executes the knapsack algorithm trying to maximise predicted winshare

    maxSalary = round((maxSalary)/100000) # maximum value the salary weight can be in knapsack

    minWinshare = players["Prediction"].min() # minimum winshare in the df

    n = players.shape[0]

    # Create a 3D array to store the results of knapsack
    dp = [[[minWinshare for _ in range(round(maxTradeValue) + 1)] for _ in range(maxSalary + 1)] for _ in range(n + 1)]
    
    #min winshare is set as inital value due to this being impossible final result of the knapsack due to its small value

    dp = np.array(dp)

    print(dp.shape)
    for i in range(1,n+1):

        #calculates knapsack

        print("i = " + str(i) + "/" + str(dp.shape[0] - 1))

        tradeValue =players.iloc[i-1]["TradeValue"]

        salary = players.iloc[i-1]["Salary"]/100000

        winshare = players.iloc[i-1]["Prediction"]

        for j in range(1,maxSalary+1):
            for k in range(1,round(maxTradeValue)+ 1):

                if salary >= j and salary < j + (salary*0.25) and tradeValue  <= k: # greedy rule of knapsack
                    
                    # adds twenty-five percent to salary weight due to salary matching rule
                    # player fits in knapsack
                    dp[i,j,k] = max(dp[i - 1, round(j - salary), round(k - tradeValue)] + winshare, dp[i -1, j, k])
                    #choose knapsack with largest winshare result

                elif salary <= j:

                    # player does not fit in the knapsack

                    dp[i,j,k] = dp[i -1, j, k]

    
    return dp

def remove_unneeded_players(players, max): 
    
    # removes players with the same trade value and salary so that only one player remains with the best result

    minTradeValue = players["TradeValue"].min()
    delete = []
    for index1, player1 in players.iterrows():
        for index2, player2 in players.iterrows():
            if index1 != index2 and round(player1["Salary"]/100000) == round(player2["Salary"]/100000) and round(player1["TradeValue"]- minTradeValue) == round(player2["TradeValue"]- minTradeValue):
                if max and player1["Prediction"] > player2["Prediction"]:
                    delete.append(index2)
                elif max == False and player1["Prediction"] < player2["Prediction"]:
                    delete.append(index1)

    players = players.drop(delete)
    return players


def min_knapsack(players, maxSalary, maxTradeValue):

    # executes the knapsack algorithm trying to minimise predicted winshare

    maxSalary = round((maxSalary)/100000)
    n = players.shape[0]
    maxWinshare = players["Prediction"].max()

    # Create a 3D array to store the results of subproblems
    intitial_dp = [[[float("inf") for _ in range(round(maxTradeValue) + 1)] for _ in range(maxSalary + 1)] for _ in range(n + 1)]
    # values intialised at infinity as we are trying to minimise result

    dp = np.array(intitial_dp)

    for i in range(n+ 1):
        for j in range(maxSalary + 1):
            dp[i,j,0] = 0
        for j in range(round(maxTradeValue) + 1):
            dp[i,0,j] = 0

    
    print(dp.shape)
    for i in range(1,n+1): # min knapsack works identically to max salary only minimising result
        print("player i = " + str(i))
        tradeValue =players.iloc[i-1]["TradeValue"]
        salary = players.iloc[i-1]["Salary"]/100000
        winshare = players.iloc[i-1]["Prediction"]
        for j in range(1,maxSalary+1):
            for k in range(1,round(maxTradeValue)+ 1):
                if salary >= j and salary < j + (salary*0.25) and tradeValue  >= k:

                    dp[i,j,k] = min(dp[i - 1, round(j - salary), round(k - tradeValue)] + winshare, dp[i -1, j, k])
                elif salary <= j:
                    dp[i,j,k] = dp[i -1, j, k]
        count = 0
    selectedIn = []
    return (dp)

def combine_knapsacks(dpOut, dpIn, attainable, teamPlayers):
    

    dim = dpIn.shape
    maxWin = 0
    maxj = 0
    maxk = 0
    for j in range(dim[1]):
        for k in range(dim[2]):
            min = np.min(dpOut[:, j, k])
            max = np.max(dpIn[:, j, k])
            current = max - min
            if current > maxWin:
                print(maxWin)
                maxWin = current 
                maxj = j
                maxk = k
                maxMin = min
                maxMax = max
    selectedIn = []
    selectedOut = []
    orgj = maxj
    orgk = maxk
    print(maxj)
    print(maxk)
    maxi = np.argmax(dpIn[:, maxj, maxk])
    while maxi > 0 and maxk > 0:
        if dpIn[maxi][maxj][maxk] != dpIn[maxi - 1][maxj][maxk]:
            player = attainable.iloc[maxi - 1]
            print(player["PLAYER"])
            selectedIn.append(player["PLAYER"])
            maxj = maxj - round(player["Salary"]/100000)
            print("newSalary")
            print(maxj)
            maxk = round(maxk - player["TradeValue"])
            print("newValue")
            print(maxk)
        maxi -= 1
    maxOuti = np.argmin(dpOut[:,orgj,orgk])
    print(maxOuti)
    print(orgj)
    print(orgk)
    print(dpOut[:,orgj,orgk])
    print(teamPlayers)
    while maxOuti > 0 and orgk > 0:
        if dpOut[maxOuti][orgj][orgk] != dpOut[maxOuti - 1][orgj][orgk]:
            selectedOut.append(teamPlayers.iloc[maxOuti - 1]["PLAYER"])
            orgj = round(orgj - teamPlayers.iloc[maxOuti - 1]["Salary"]/100000)
            orgk = round(orgk - teamPlayers.iloc[maxOuti - 1]["TradeValue"])
        maxOuti -= 1
    return [selectedIn, selectedOut, maxWin]

def Defensive_pos(FGA, TO, FTA, OR):
    # calculates defensive possessions a team faces

    possession_formula = 0.96 * (FGA + TO + 0.44 * FTA - OR)
    return possession_formula

def calculate_pp(fga, fta, to, offensive_rating , possessions, gp):

    # calculates a player's points produceds

    points_produced = (offensive_rating/100) * possessions
    return points_produced



def calculate_ScPoss(FGM, PTS, FTM, FGA, MP, Team_MP, Team_PTS,Team_AST, AST, Team_FGM, Team_FTM, Team_FGA, Team_TOV, FTA, Team_FTA, Team_ORB, ORB, Team_ORB_Percentage):

    # calculates scoring possession as outlined in project specification

    qAST = ((MP / (Team_MP)) * (1.14 * ((Team_AST - AST) / Team_FGM))) + ((((Team_AST / Team_MP) * MP - AST) / ((Team_FGM / Team_MP) * MP - FGM)) * (1 - (MP / (Team_MP))))
    AST_Part = 0.5 * (((Team_PTS - Team_FTM) - (PTS - FTM)) / (2 * (Team_FGA - FGA))) * AST
    FT_Part = 0
    if FTA > 0:
        FT_Part = (1 - (1 - (FTM / FTA))**2) * 0.4 * FTA
    Team_Scoring_Poss = Team_FGM + (1 - (1 - (Team_FTM / Team_FTA))**2) * Team_FTA * 0.4
    Team_Play_Percentage = Team_Scoring_Poss / (Team_FGA + Team_FTA * 0.4 + Team_TOV)
    Team_ORB_Weight = ((1 - Team_ORB_Percentage) * Team_Play_Percentage) / ((1 - Team_ORB_Percentage) * Team_Play_Percentage + Team_ORB_Percentage * (1 - Team_Play_Percentage))
    ORB_Part = ORB * Team_ORB_Weight * Team_Play_Percentage
    
    FG_Part = FGM * (1 - 0.5 * ((PTS - FTM) / (2 * FGA)) * qAST)
    ScPoss = (FG_Part + AST_Part + FT_Part) * (1 - (Team_ORB / Team_Scoring_Poss) * Team_ORB_Weight * Team_Play_Percentage) + ORB_Part
    return ScPoss


def calculate_FGxPoss(FGA, FGM, Team_ORB_Percentage):

    # calculates field goal possessions

    FGxPoss = (FGA - FGM) * (1 - 1.07 * Team_ORB_Percentage)
    return FGxPoss

def calculate_FTxPoss(FTM, FTA):

    # calculates free throw possessions

    FTxPoss = 0
    if FTA > 0:
        FTxPoss = ((1 - (FTM / FTA))**2) * 0.4 * FTA 
    return FTxPoss

def offensive_poss(Team_MP, FGM, PTS, FTM, FGA, GP, Team_PTS,Team_AST, AST, Team_FGM, Team_FTM, Team_FGA, Team_TOV, FTA, Team_FTA, Team_ORB, ORB, Team_ORB_Percentage):

    #calculates the offensive possessions of a player 

    return (
        calculate_ScPoss(GP * FGM, GP * PTS, GP * FTM, GP * FGA, GP,Team_MP, Team_MP * Team_PTS,Team_MP * Team_AST, GP * AST, Team_MP * Team_FGM, Team_MP * Team_FTM, Team_MP * Team_FGA, Team_MP * Team_TOV, GP * FTA, Team_MP * Team_FTA, Team_MP * Team_ORB, GP * ORB, Team_ORB_Percentage) 
        + calculate_FGxPoss(FGA*GP, FGM*GP, Team_ORB_Percentage/100) 
        + calculate_FTxPoss(FTM*GP, FTA*GP)
    )

def calculate_basketball_production(Team_MP, FGM, _3PM, PTS, FTM, FGA, AST, Team_FGM, Team_3PM, Team_PTS, Team_FTM, Team_FTA, Team_FGA, ORB, Team_ORB, Team_AST, MP, Team_ORB_Percentage, Team_TOV):
    qAST = ((MP / (Team_MP)) * (1.14 * ((Team_AST - AST) / Team_FGM))) + ((((Team_AST / Team_MP) * MP - AST) / ((Team_FGM / Team_MP) * MP - FGM)) * (1 - (MP / (Team_MP))))
    Team_Scoring_Poss = Team_FGM + (1 - (1 - (Team_FTM / Team_FTA))**2) * Team_FTA * 0.4
    Team_Play = Team_Scoring_Poss / (Team_FGA + Team_FTA * 0.4 + Team_TOV)
    Team_ORB_Weight = ((1 - Team_ORB_Percentage) * Team_Play) / ((1 - Team_ORB_Percentage) * Team_Play + Team_ORB_Percentage * (1 - Team_Play))
    PProd_FG_Part = 2 * (FGM + 0.5 * _3PM) * (1 - 0.5 * ((PTS - FTM) / (2 * FGA)) * qAST)
    PProd_AST_Part = 2 * ((Team_FGM - FGM + 0.5 * (Team_3PM - _3PM)) / (Team_FGM - FGM)) * 0.5 * (((Team_PTS - Team_FTM) - (PTS - FTM)) / (2 * (Team_FGA - FGA))) * AST
    PProd_ORB_Part = ORB * Team_ORB_Weight * Team_Play * (Team_PTS / (Team_FGM + (1 - (1 - (Team_FTM / Team_FTA))**2) * 0.4 * Team_FTA))
    PProd = (PProd_FG_Part + PProd_AST_Part + FTM) * (1 - (Team_ORB / Team_Scoring_Poss) * Team_ORB_Weight * Team_Play) + PProd_ORB_Part

    return PProd

def name_to_approx_value(player):
    # uses stats points, rebounds, assists, steals, blocks, field_goals_missed, free_throws_missed, turnovers
    return approx_value(player["Pts"], player["REB"], player["AST"], player["STL"], player["BLK"], player["FGA"] - player["FGM"], player["FTA"] - player["FTM"], player["TOV"])


def adjust_tv(players):
    
    # adjusts trade value so the minimum value is 0 for the knapsack

    minTradeValue = players["TradeValue"].min()
    for index, _ in players.iterrows():
        players.at[index, "TradeValue"] -= minTradeValue
    return players


def recommend(team, untradeable):
    
    # combines knapsack algorithms to find best trade
    intialcurrentPlayers = players_df[players_df["Year"] == current_year - 1]
    currentPlayers = intialcurrentPlayers.copy()
    value = 0 
    salary = 0
    winshare_range(currentPlayers, team)
    currentPlayers = currentPlayers[currentPlayers["Prediction"].notna()]
    approx_value_range(currentPlayers)
    currentPlayers = adjust_tv(currentPlayers)
    attainableAssets = currentPlayers[(currentPlayers['TEAM'] != team)]
    currentAssets = currentPlayers[(currentPlayers['TEAM'] == team) & (~players_df["PLAYER"].isin(untradeable))]
    maxSal = sum_salary(currentAssets)
    maxVal = sum_approx_value(currentAssets)
    attainableAssets = remove_unneeded_players(attainableAssets, True)
    currentAssets = remove_unneeded_players(currentAssets, False)
    dpIn = max_knapsack(attainableAssets, maxSal, maxVal)
    dpOut = min_knapsack(currentAssets, maxSal, maxVal)
    result = combine_knapsacks(dpOut, dpIn, attainableAssets, currentAssets)
    inmax = 0
    outmax = 0
    for player in result[0]:
        inmax += attainableAssets[attainableAssets["PLAYER"] == player]["Prediction"].values[0]
    for player in result[1]:
        outmax += currentAssets[currentAssets["PLAYER"] == player]["Prediction"].values[0]
    return [result[0], result[1], inmax - outmax]

def approx_value_range(players):
    players['TradeValue'] = players.apply(lambda row: name_to_approx_value(row), axis=1)
    return 0

def name_to_predict_win(name, team):
    return predict(name,team)

def winshare_range(players, team):
    players['Prediction'] = players.apply(lambda row: name_to_predict_win(row["PLAYER"], team), axis=1)
    return 0

def sum_salary(players): # finds the total aggregate salaires of the top n amount of highest earners
    salary = 0
    salaries = np.array(players["Salary"])
    if len(salaries == 1):
        biggest_salaries = salaries
    else:
        top_indices = np.argpartition(salaries, (-maxOut))[-maxOut:]
        biggest_salaries = salaries[top_indices]
    for number in biggest_salaries:
        salary += number
    return salary

def sum_approx_value(players): # finds the total aggregate approx value of the top n amount of highest value players
    maxValue = 0
    salaries = np.array(players["TradeValue"])
    if len(salaries == 1):
        biggest_values = salaries
    else:
        top_indices = np.argpartition(salaries, (-maxOut))[-maxOut:]
        biggest_values = salaries[top_indices]
    for number in biggest_values:
        maxValue += number
    return maxValue


def accuracy_measure(data):
    
    #calculates mean absolute error, mean square error and percentage of error of system over the entire players df
    current_year = 1
    winshare_total = 0
    difference = 0
    toPredict = data[(data["Year"] == 2024)]
    oldPlayers = data[data["Year"] < 2024]
    total_predict =0 
    square_total = 0
    i = 0
    j = 0
    k = 0
    outliers = np.zeros(36)
    avgover = 0
    avgunder = 0
    mega_out = 0
    for index, player in toPredict.iterrows():
        if not oldPlayers[oldPlayers["PLAYER"] == player["PLAYER"]].empty:
            i+=1
            temp = predict(player["PLAYER"], player["TEAM"])
            actual = player["Winshare"]
            error = round((temp - actual)) + 18
            outliers[error] += 1
            if abs(temp - actual) > 4:
                print(player["PLAYER"])
                print(actual)
                print(temp)
                mega_out +=1 

            winshare_total += actual
            difference += abs(temp - actual)
            total_predict += temp
            square_total += (actual - temp)**2
            if (temp - actual > 0):
                j+= 1
                avgover += player["AGE"]
            else:
                k+=1
                avgunder += player["AGE"]
    mae = difference/i
    percentError = total_predict / winshare_total
    mse = square_total/i
    return [mae, percentError, mse]

def predict(player_name, new_team):

    #gets the prediction of the winshare of a given player on a given team from Y

    if player_name in collab_data['PLAYER'].values:
        players = collab_data['PLAYER'].unique().tolist()
        players.sort() 
        player = players.index(player_name)
        team = team_df.index[(team_df["ABR"] == new_team) & (team_df["Year"] == 2023)].tolist()[0]
        return Y[team, player]
    return np.nan


def regression_Z(training_data, p):

    # calculates the polynomial regression of polynomial p

    n = len(training_data)
    Z = np.zeros((n,p))
    W = np.ones((1, n))

    training_data = training_data.groupby('PLAYER').filter(lambda x: len(x) > 3)
    training_data["Winshare"] = training_data["Winshare"] - training_data["Winshare"].min()
    X = training_data['AGE'].values.reshape(-1, 1)
    y = training_data['Winshare'].values.reshape(-1, 1)

    # Polynomial features transformation
    poly_features = PolynomialFeatures(degree=p, include_bias=False)
    X_poly = poly_features.fit_transform(X) # creates a n-by-p matrix of the age data for polynomial regression

    # Fit linear regression model
    lin_reg = LinearRegression() 
    lin_reg.fit(X_poly, y) # minimises the loss of data using a polynomial curve of degree p

    # Visualise the results
    X_new = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly) 

    # plt.scatter(X, y, label='Original Data') #This plots the linear regression uncomment code to seet
    # plt.plot(X_new, y_new, 'r-', label='Polynomial Regression')
    # plt.xlabel('Age')
    # plt.ylabel('Points Scored')
    # plt.legend()
    # plt.title('Polynomial Regression Example')
    # plt.show()
    return X_new, y_new


team_df = pd.read_csv(team_path)
team_df = team_df.sort_values(by="Year", ascending=True)
team_df = team_df.reset_index(drop=True) # load in the team data and sort the rows by year
players_df = pd.read_csv(players_path)
players_df = players_df[players_df["GP"] > 5] # players who have played 5 games or less are removed from the database due to the lack of substatianal data
players_df = players_df.sort_values(by="Year", ascending=True)
players_df = players_df.reset_index(drop=True) # sort player csv by year
players_df["Salary"] = pd.to_numeric(players_df["Salary"].str.replace(r'[$,]', ''), errors='coerce') # convet salary data from a string to an integer
players_df["Winshare"] = players_df.apply(lambda row: winshare(row, row["Year"]), axis=1) #calculate and stores the winshare of every player in the database
testData = pd.read_csv(team_data_path) #loads in a list of players who changed team befor the start of the 2023/24 season
player_training_data = players_df[players_df["Year"] < current_year - 1]
collab_data = player_training_data.groupby('PLAYER').agg({'AGE': 'max', "Pts": 'mean', "AST": 'mean', "FG%": 'mean', "OREB%": 'mean', "TOV": 'mean', "FT%": 'mean', "FGA": 'mean', "OREB": 'mean', "3P%": 'mean', "3PM": 'mean', "BLK": 'mean', "MIN": 'mean', "DEFRTG": 'mean', "DREB": 'mean', "DREB%": 'mean', "Year": 'max' }).reset_index()
Y = np.load('array_data.npy')

def train(players_df, team_df):

    player_training_data = players_df[players_df["Year"] < current_year - 1]
    team_training_data = team_df[team_df["Year"] < current_year - 1]
    min_winshare = player_training_data["Winshare"].min()
    age_reg, performance_reg = regression_Z(player_training_data, 6) # calculates regression of the system
    # calulates the mean and max of player data for the item feature matrix later

    Y = train_collab(100,0.0009, get_item_matrix(collab_data), get_user_matrix(team_training_data), player_training_data, team_training_data, min_winshare, age_reg, performance_reg)

    toTest = players_df[players_df["Year"] < current_year]

    print(accuracy_measure(toTest)) #calculate accuracy of whole system
    print(accuracy_measure(toTest[toTest["PLAYER"].isin(testData["PLAYER"].values)])) # calculates the accuracy of whole system

if __name__ == "__main__":
    train(players_df, team_df)
