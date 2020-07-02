import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime

# de_para_siglas = pd.read_excel("de_para_siglas_br.xlsx")
# de_para_siglas["time"] = de_para_siglas["time"].astype(str)
# de_para_siglas["time"] = de_para_siglas["time"].apply(str.strip)
# de_para_siglas.set_index("time", inplace=True)

def parse_event_info(players_df, de_para_siglas, player_i = 0):
    try:
        jogo = players_df.iloc[player_i].eventData
    except IndexError:
        return {}

    resp = {}
    
    resp["team_away"] = de_para_siglas.loc[jogo["awayTeam"]["name"]].iloc[0]
    resp["team_home"] = de_para_siglas.loc[jogo["homeTeam"]["name"]].iloc[0]
    resp["data"] = datetime.utcfromtimestamp(jogo["startTimestamp"] - 7200).strftime('%Y-%m-%d')
    resp["hora"] = datetime.utcfromtimestamp(jogo["startTimestamp"] - 7200).strftime('%H:%M:%S')
    resp["game"] = resp["team_home"] + " X " + resp["team_away"] + " " + resp["data"]
    resp["away_score"] = jogo["awayScore"]
    resp["home_score"] = jogo["homeScore"]
    if resp["away_score"] > resp["home_score"]:
        resp["result"] = -1
    elif resp["away_score"] == resp["home_score"]:
        resp["result"] = 0
    else:
        resp["result"] = 1
    return (resp)

def parse_player_info(players_df, player_i):
    resp = dict(players_df.iloc[player_i].player)
    
    team_info = dict(players_df.iloc[player_i].team)
    team_info["team_name"] = team_info["name"]
    team_info["team_id"] = team_info["id"]
    del team_info["name"]
    del team_info["id"]
    del team_info["gender"]
    
    resp.update(team_info)
    return (resp)

def parse_info(players_df, player_i, info_kind="summary"):
    try:
        return(pd.DataFrame(players_df.iloc[player_i].groups[info_kind]["items"]).loc["value"])
    except TypeError:
        raise TypeError
        
def parse_all_info(players_df, player_i):
    resp = parse_player_info(players_df, player_i)
    for info in players_df.iloc[player_i].groups.keys():
        try:
            r = parse_info(players_df, player_i, info)
            resp.update(r)
        except TypeError:
            pass
        
    for key in resp.keys():
        if resp[key] == []:
            resp[key] = np.nan
    
    del resp["notes"]
    
    return(resp)

def parse_all_info_all_players(players_df):
    resp = []
    for i in range(len(players_df)):
        resp.append(pd.DataFrame(parse_all_info(players_df, i), index=[0]))
    return(pd.concat(resp, sort=False).reset_index())


def get_per_player_data(players_df, de_para_siglas):
    game_info = parse_event_info(players_df, de_para_siglas)
    players_data = parse_all_info_all_players(players_df).drop("index", axis=1)
    
    players_data["game"] = [game_info["game"] for i in range(len(players_data))]
    players_data["team"] = [de_para_siglas.loc[x].iloc[0] for x in players_data.team_name]
    return(players_data)
    
def get_odds(resp, map_odds = {0: "final_result",
                               1: "double_chance",
                               2: "first_half",
                               3: "draw_no_bet",
                               4: "both_score",
                               5: "total_goals"}):
    ret = {}
    
    
    odds = resp.json()["odds"]
    for key in map_odds:
        if key < 5:
            try:
                for possible_result in odds[key]["regular"][0]["odds"]:
                    ret[map_odds[key] + "_" + str(possible_result["choice"])] = possible_result["decimalValue"] 
                    ret["fl_" + map_odds[key] + "_" + str(possible_result["choice"])] = possible_result["winning"]
            except IndexError:
                return ret
        else:
            goals_over_under = ["0.5", "1.5", "2.5", "3.5", "4.5", "5.5"]
            for num_goals in range(len(goals_over_under)):
                try:
                    for possible_result in odds[key]["regular"][num_goals]["odds"]:
                        ret[map_odds[key] + "_" + str(possible_result["choice"]) + "_" + goals_over_under[num_goals]] = possible_result["decimalValue"] 
                        ret["fl_" + map_odds[key] + "_" + str(possible_result["choice"]) + "_" + goals_over_under[num_goals]] = possible_result["winning"]
                except IndexError:
                    ret[map_odds[key] + "_" + str(possible_result["choice"]) + "_" + goals_over_under[num_goals]] = -1
                    ret["fl_" + map_odds[key] + "_" + str(possible_result["choice"]) + "_" + goals_over_under[num_goals]] = False
    return(ret)

def get_live_form(resp, prefix = "form_minute_"):
    ret = {}
    try:
        for line in resp.json()["liveForm"]:
            ret[prefix + str(line["minute"])] = line["value"]
    except TypeError:
        pass
    return(ret)

def game_statistics(resp, players_df, de_para_siglas, periods = [0, 1, 2]):
    if players_df is not None:
        ret = parse_event_info(players_df, de_para_siglas)
    else:
        ret = {}

    
    if resp.json()["statistics"] is not None:
        for period in periods:
            period_suffix = resp.json()["statistics"]["periods"][period]['period']
            groups = resp.json()["statistics"]["periods"][period]["groups"]
            for i in range(len(groups)):
                items = groups[i]["statisticsItems"]
                for item in items:
                    nome = item["name"]
                    ret[nome + "_home_" + period_suffix] = item["home"]
                    ret[nome + "_away_" + period_suffix] = item["away"]
        
    ret.update(get_live_form(resp))
    try:
        ret.update(get_odds(resp))
    except KeyError:
        pass
    
    return(ret)

def get_info_rodada(resp, de_para_siglas):
    games = resp.json()['roundMatches']["tournaments"][0]['events']
    
    resp = {
        "game":[],
        "fixture": [],
        "id":[],
        "team_home":[],
        "team_away":[],
        "date":[],
        "link":[],
        "home_score":[],
        "away_score":[]
    }
    
    for game in games:
        resp["fixture"].append(game['roundInfo']["round"])
        resp["id"].append(game["id"])
        #print(game["homeTeam"]["name"], de_para_siglas.loc[game["homeTeam"]["name"]].iloc[0])
        #print(game["awayTeam"]["name"], de_para_siglas.loc[game["awayTeam"]["name"]].iloc[0])
        
        resp["team_home"].append(de_para_siglas.loc[game["homeTeam"]["name"]].iloc[0])
        resp["team_away"].append(de_para_siglas.loc[game["awayTeam"]["name"]].iloc[0])
        
        try:        
            resp["home_score"].append(game['homeScore']["current"])
            resp["away_score"].append(game['awayScore']["current"])        
        except (TypeError, KeyError):
            resp["home_score"].append(-1)
            resp["away_score"].append(-1)
        resp["date"].append(game["formatedStartDate"].replace(".", "-")[:-1])
        resp["link"].append("https://www.sofascore.com/pt/" + game["slug"] + "/" + game["customId"])
        resp["game"].append(resp["team_home"][-1] + " X " + resp["team_away"][-1] + " " + resp["date"][-1])
    
    return(pd.DataFrame(resp).set_index("game"))

def get_incidents_database(
        resp_incidents,
        types=['period', 'substitution', 'injuryTime', 'goal', 'card', 'varDecision']
    ):

    incidents_per_type = {}

    r_json = resp_incidents.json()
    if 'incidents' in r_json:
        for el in r_json["incidents"]:
            if el['incidentType'] in types:
                if not el['incidentType'] in incidents_per_type:
                    incidents_per_type[el['incidentType']] = []
                incidents_per_type[el['incidentType']].append(el)
    
        for key in incidents_per_type:
            incidents_per_type[key] = pd.DataFrame(incidents_per_type[key])
    
    return incidents_per_type