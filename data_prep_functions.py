import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import geopy.distance
import gc
from tqdm import tqdm_notebook

def get_last_games(df, data, team_name, n = 5, filter="all", verbose=False):
    """
        Retorna os últimos n jogos de um determinado time na visão Home ou Away
        
        Parâmetros: 
                   df: DataFrame com os dados individuais dos jogos já ocorridos
                 data: Data de referência para o filtro dos últimos jogos
            team_name: Nome do time que se busca
                    n: Tamanho da janela dos últimos jogos que se deseja ver
               filter: Pode ser 'home', 'away' ou 'all'
              verbose: Boolean
    """
    
    if(filter == "all"):
        last_games = df[(df["data"] < data) & 
                        ((df["team_home"] == team_name) | (df["team_away"] == team_name))].tail(n)
        
    elif(filter == "home"):
        last_games = df[(df["data"] < data) & (df["team_home"] == team_name)].tail(n)
    elif(filter == "away"):
        last_games = df[(df["data"] < data) & (df["team_away"] == team_name)].tail(n)

    if(verbose):
        print(last_games[["team_home", "team_away", "PTS_home", "PTS_away", "DATE"]])
    
    if not isinstance(last_games, pd.DataFrame):
        last_games = last_games.to_frame().transpose()
        
    return(last_games)

def get_avg_last_games(last_games, team_name, home_columns, away_columns,
                       n = 5, data_ref = None, rivals = False, fl_result="fl_home_win",
                       to_drop=['TEAM_NAME_misc_away', 'GAME', 'TEAM_NAME_playertrack_home', 
                                'fl_win', 'fl_home_win', 'TEAM_NAME_playertrack_away', 'TEAM_NAME_scoring_home', 
                                'team_game_num', 'GAME_DATE_home', 'SEASON', 'TEAM_NAME_advanced_away', 
                                'TEAM_CITY_home', 'fl_playoff', 'index', 'TEAM_NAME_misc_home', 'TEAM_ID_away', 
                                'TEAM_NAME_usage_home', 'TEAM_CITY_away', 'TEAM_NAME_home', 'TEAM_NAME_fourfactors_away', 
                                'TEAM_NAME_advanced_home', 'GAME_ID_home', 'TEAM_NAME_hustle_away', 'team_away_game_num',
                                'team_home_game_num', 'TEAM_NAME_hustle_home', 'GAME_ID_away', 'TEAM_NAME_away', 'TEAM_NAME_usage_away',
                                'TEAM_ID_home', 'DATE', 'GAME_PLACE_home', 'TEAM_NAME_fourfactors_home', 'TEAM_NAME_scoring_away']):
    """
        Retorna a média do desempenho do time (ou seus rivais) nos últimos N jogos
        
        Parâmetros:
            last_games: DataFrame com as linhas individuais dos dados dos últimos N jogos já filtrados pelo método 'get_last_games()'
             team_name: Nome do time que se busca sumarizar os jogos
          home_columns: Lista com as colunas referentes ao desempenho do time mandante
          away_columns: Lista com as colunas referentes ao desempenho do time visitante
                     n: Tamanho da janela dos últimos jogos que se deseja ver
              data_ref: Data de referência para o filtro dos últimos jogos
                rivals: (Boolean) True para retornar o desempenho médio dos rivais do time na janela específica
               to_drop: (List) Colunas a desconsiderar
    """
    
    if(rivals == False):
        last_games_home = last_games[last_games["team_home"] == team_name].sum().to_frame().transpose().drop(away_columns + to_drop + ["team_home"], axis=1, errors="ignore")
        last_games_away = last_games[last_games["team_away"] == team_name].sum().to_frame().transpose().drop(home_columns + to_drop + ["team_away"], axis=1, errors="ignore")
        
        last_games_home.columns = [x.replace("_home","") for x in last_games_home.columns]
        last_games_away.columns = [x.replace("_away","") for x in last_games_away.columns]
    else:
        last_games_home = last_games[last_games["team_home"] != team_name].sum().to_frame().transpose().drop(away_columns + to_drop + ["team_home"], axis=1, errors="ignore")
        last_games_away = last_games[last_games["team_away"] != team_name].sum().to_frame().transpose().drop(home_columns + to_drop + ["team_away"], axis=1, errors="ignore")
            
        last_games_home.columns = [x.replace("_home","_opponent") for x in last_games_home.columns]
        last_games_away.columns = [x.replace("_away","_opponent") for x in last_games_away.columns]
    
    if(n == 10000 or n <= (len(last_games_home) + len(last_games_away))):
        n = len(last_games_home) + len(last_games_away)       
    
    if(len(last_games_home) == 0):
        if(rivals):
            resp = (last_games_away/n)
        else:
            resp = last_games_away/n
    
    if(len(last_games_away) == 0):
        if(rivals):
            resp = (last_games_home/n)
        else:
            resp = last_games_home/n
    
    if(len(last_games_away) > 0 and len(last_games_home) > 0):
        if(rivals):
            resp = ((last_games_home + last_games_away) / n)
        else:
            resp = ((last_games_home + last_games_away) / n)    
    
    if not rivals:
        var_criadas = cria_variaveis_sumarizacao(last_games, team_name, n = 5, data_ref = data_ref)
        resp = pd.concat([resp, var_criadas], axis=1).sum().to_frame().transpose()
    
    return(resp)

def get_season(date):
    """
        Retorna a temporada referente a uma determinada data
        
        Parâmetros:
            date: Data de um jogo
    """
    ano = date.year
    if(date.month >= 10):
        return ano + 1
    return ano

def is_playoff(date):
    """
        Retorna 1 se uma data faz parte de um intervalo de jogos de playoffs ou 0 em caso contrário
        
        Parâmetros:
            date: Data de um jogo 
    """
    # Playoffs 2016
    if date >= datetime(2016, 4, 16) and date < datetime(2016, 6, 30):
        return 1

    # Playoffs 2017
    elif date >= datetime(2017, 4, 15) and date < datetime(2017, 6, 30):
        return 1

     # Playoffs 2018
    elif date >= datetime(2018, 4, 14) and date < datetime(2018, 6, 30):
        return 1

    return 0

def get_dist_last_game(df, data, df_dist, team_home, team_away, is_home=True):    
    """
        Retorna a distância em KM percorrida por um time específico para chegar a um jogo
        
        Parâmetros:
            df: DataFrame com os dados individuais dos jogos já ocorridos
          data: Data do jogo em referência
       df_dist: DataFrame com a matriz de distâncias entre todos os times
     team_home: Time da Casa no jogo em referência
     team_away: Time Visitante no jogo em referência
       is_home: (Boolean) Distância na visão do time da casa (True) ou no time visitante (False)
    """

    if(is_home):
        last_game = get_last_games(df, data, team_home, n = 1)

        if(len(last_game) == 0):
            return(0)

        if (last_game.team_home.iloc[0] == team_home):
            return(0)
        else:
            return(df_dist.loc[team_home, last_game.team_home.iloc[0]])        
    else:
        last_game = get_last_games(df, data, team_away, n = 1)

        if(len(last_game) == 0):
            return(df_dist.loc[team_home, team_away])

        if (last_game.team_away.iloc[0] == team_away):
            return(df_dist.loc[team_home, last_game.team_away.iloc[0]])
        else:
            return(df_dist.loc[team_home, team_away])

def get_days_from_last_game(df, data, team_name):    
    """
        Retorna o número de dias entre o jogo atual e o jogo passado
        
        Parâmetros:
            df: DataFrame com os dados individuais dos jogos já ocorridos
          data: Data do jogo em referência
     team_name: Nome do time em referência
    """

    last_game = get_last_games(df, data, team_name, n = 1)

    if(len(last_game) == 0):
        return(np.nan)

    return(-(last_game["data"] - data).iloc[0].days)

def cria_variaveis_sumarizacao(last_games, team_name, n = 5, data_ref = None, verbose = False):
    """
        Cria variáveis extras ao sumarizar os últimos jogos no método 'get_avg_last_games()'.
        As variáveis novas incluem: Win %, N_Games e Days_Diff nos cenários away e home
    
        Parâmetros:
            last_games: DataFrame com as linhas individuais dos dados dos últimos N jogos já filtrados pelo método 'get_last_games()'
             team_name: Nome do time em referência
                     n: Tamanho da janela de últimos jogos
              data_ref: Data de referência  
    """
    
    resp = {}
    
    # -------------------------
    # Cria variávies de Win %
    # -------------------------
    # Visão home
    resp["N_WINS_HOME"] = [np.where((last_games["team_home"] == team_name) &
                             (last_games["fl_home_win"] == 1) , 1, 0).sum()]
    resp["N_GAMES_HOME"] = [np.where(last_games["team_home"] == team_name, 1, 0).sum()]
    resp["WIN_HOME_PCT"] = [(resp["N_WINS_HOME"][0] / resp["N_GAMES_HOME"][0])]
    
    # Visão Away
    resp["N_WINS_AWAY"] = [np.where((last_games["team_away"] == team_name) & 
                                      (last_games["fl_home_win"] == 0), 1, 0).sum()]
    resp["N_GAMES_AWAY"] = [np.where(last_games["team_away"] == team_name, 1, 0).sum()]
    resp["WIN_AWAY_PCT"] = [(resp["N_WINS_AWAY"][0] / resp["N_GAMES_AWAY"][0])]
    
    # Visão geral
    resp["N_WINS_TOTAL"] = [resp["N_WINS_AWAY"][0] + resp["N_WINS_HOME"][0]]
    resp["WIN_PCT"] = [resp["N_WINS_TOTAL"][0]/(resp["N_GAMES_AWAY"][0] + resp["N_GAMES_HOME"][0])]
    # -------------------------
    
    # -------------------------
    # Cria variáveis de Draw %
    # -------------------------
    if "fl_draw" in last_games.columns:
        # Visão home
        resp["N_WINS_HOME"] = [np.where((last_games["team_home"] == team_name) &
                                 (last_games["fl_draw"] == 1) , 1, 0).sum()]
        resp["N_GAMES_HOME"] = [np.where(last_games["team_home"] == team_name, 1, 0).sum()]
        resp["WIN_HOME_PCT"] = [(resp["N_WINS_HOME"][0] / resp["N_GAMES_HOME"][0])]

        # Visão Away
        resp["N_WINS_AWAY"] = [np.where((last_games["team_away"] == team_name) & 
                                          (last_games["fl_draw"] == 0), 1, 0).sum()]
        resp["N_GAMES_AWAY"] = [np.where(last_games["team_away"] == team_name, 1, 0).sum()]
        resp["WIN_AWAY_PCT"] = [(resp["N_WINS_AWAY"][0] / resp["N_GAMES_AWAY"][0])]

        # Visão geral
        resp["N_WINS_TOTAL"] = [resp["N_WINS_AWAY"][0] + resp["N_WINS_HOME"][0]]
        resp["WIN_PCT"] = [resp["N_WINS_TOTAL"][0]/(resp["N_GAMES_AWAY"][0] + resp["N_GAMES_HOME"][0])]
    # -------------------------
    
    if verbose:
        print("Win_PCT", resp["WIN_PCT"][0], resp["N_WINS_AWAY"][0], resp["N_WINS_HOME"][0])
    
    # Cria variáveis de data
    if(data_ref is None):
        data_ref = np.max(last_games["data"]) + timedelta(days=1)
    
    # Visão da série
    resp["TOTAL_DAYS_DIFF"] = [(np.max(last_games["data"]) - np.min(last_games["data"])).days]
    days_diff_last_games = [-x.days if not np.isnan(x.days) else 0 
                            for x in last_games["data"].sub(last_games["data"].shift(-1).fillna(data_ref))]
    resp["DAYS_DIFF_LG_STD"] = [np.std(days_diff_last_games)]
    resp["DAYS_DIFF_LG_MEAN"] = [np.mean(days_diff_last_games)]
    
    if verbose:
        print("Days_Diff", days_diff_last_games, resp["DAYS_DIFF_LG_MEAN"][0], resp["TOTAL_DAYS_DIFF"][0]) 
    
    # ---------------------
    # All
    # ---------------------
    resp["N_GAMES_L2_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=2))) & 
                            (last_games["data"] < data_ref) &
                            ((last_games["team_home"] == team_name) |
                             (last_games["team_away"] == team_name)), 1, 0).sum()]
    
    resp["N_GAMES_L4_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=4))) & 
                            (last_games["data"] < data_ref) &
                            ((last_games["team_home"] == team_name) |
                             (last_games["team_away"] == team_name)), 1, 0).sum()]
                             
    resp["N_GAMES_L6_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=6))) & 
                            (last_games["data"] < data_ref) &
                            ((last_games["team_home"] == team_name) |
                             (last_games["team_away"] == team_name)), 1, 0).sum()]
    
    resp["N_GAMES_L8_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=8))) & 
                            (last_games["data"] < data_ref) &
                            ((last_games["team_home"] == team_name) |
                             (last_games["team_away"] == team_name)), 1, 0).sum()]
    
    resp["N_GAMES_L10_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=10))) & 
                            (last_games["data"] < data_ref) &
                            ((last_games["team_home"] == team_name) |
                             (last_games["team_away"] == team_name)), 1, 0).sum()]
    
    # Away
    resp["N_GAMES_AWAY_L2_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=2))) & 
                                    (last_games["data"] < data_ref) &
                                    (last_games["team_away"] == team_name), 1, 0).sum()]
    
    resp["N_GAMES_AWAY_L4_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=4))) & 
                                    (last_games["data"] < data_ref) &
                                    (last_games["team_away"] == team_name), 1, 0).sum()]
    
    resp["N_GAMES_AWAY_L6_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=6))) & 
                                    (last_games["data"] < data_ref) &
                                    (last_games["team_away"] == team_name), 1, 0).sum()] 
    
    resp["N_GAMES_AWAY_L8_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=8))) & 
                                    (last_games["data"] < data_ref) &
                                    (last_games["team_away"] == team_name), 1, 0).sum()]
    
    resp["N_GAMES_AWAY_L8_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=10))) & 
                                    (last_games["data"] < data_ref) &
                                    (last_games["team_away"] == team_name), 1, 0).sum()]
    
    # Home    
    resp["N_GAMES_AWAY_L2_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=2))) & 
                                    (last_games["data"] < data_ref) &
                                    (last_games["team_home"] == team_name), 1, 0).sum()]
    
    resp["N_GAMES_AWAY_L4_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=4))) & 
                                    (last_games["data"] < data_ref) &
                                    (last_games["team_home"] == team_name), 1, 0).sum()]
    
    resp["N_GAMES_AWAY_L6_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=6))) & 
                                    (last_games["data"] < data_ref) &
                                    (last_games["team_home"] == team_name), 1, 0).sum()]
    
    resp["N_GAMES_AWAY_L8_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=8))) & 
                                    (last_games["data"] < data_ref) &
                                    (last_games["team_home"] == team_name), 1, 0).sum()]
    
    resp["N_GAMES_AWAY_L10_days"] = [np.where((last_games["data"] >= (data_ref - timedelta(days=10))) & 
                                    (last_games["data"] < data_ref) &
                                    (last_games["team_home"] == team_name), 1, 0).sum()]
    
    if verbose:
        print("Num Games Last X Days", resp["N_GAMES_L6_days"][0], resp["N_GAMES_AWAY_L6_days"][0])
        
    # -------------
    # Distance KM
    # -------------
    if "DISTANCE_KM_home" in last_games.columns:
        resp["SUM_DIST_KM"] = [last_games[last_games["team_home"] == team_name]["DISTANCE_KM_home"].sum()
                               + last_games[last_games["team_away"] == team_name]["DISTANCE_KM_away"].sum()]

        dist_list = (list(last_games[last_games["team_home"] == team_name]["DISTANCE_KM_home"])
                     + list(last_games[last_games["team_away"] == team_name]["DISTANCE_KM_away"]))

        resp["AVG_DIST_KM"] = [np.average(dist_list)]

        # Back to Back
        resp["BACK_TO_BACK"] = [np.where(pd.Series(dist_list) > 3500, 1, 0).sum()]
    
    
    # Days from last Game
    days_from_last_games_list = (list(last_games[last_games["team_home"] == team_name]["DAYS_FROM_LAST_GAME_home"])
                                 + list(last_games[last_games["team_away"] == team_name]["DAYS_FROM_LAST_GAME_away"]))
    
    if(len(days_from_last_games_list) > 0):
        resp["AVG_DAYS_FROM_LG"] = [np.average(days_from_last_games_list)]
        resp["STD_DAYS_FROM_LG"] = [np.std(days_from_last_games_list)]
        resp["MIN_DAYS_FROM_LG"] = [np.min(days_from_last_games_list)]
    else:
        resp["AVG_DAYS_FROM_LG"] = [np.nan]
        resp["STD_DAYS_FROM_LG"] = [np.nan]
        resp["MIN_DAYS_FROM_LG"] = [np.nan]
    
    
    # Dominance Variables
    resp["total_minutes_dominant"] = last_games[last_games["team_home"] == team_name]["minutes_dominant_home"].sum() + last_games[last_games["team_away"] == team_name]["minutes_dominant_away"].sum()
    
    resp["total_dominance"] = last_games[last_games["team_home"] == team_name]["total_dominance_home"].sum() + last_games[last_games["team_away"] == team_name]["total_dominance_away"].sum()
    
    resp["avg_total_minutes_dominant"] = np.mean(list(last_games[last_games["team_home"] == team_name]["minutes_dominant_home"]) +  list(last_games[last_games["team_away"] == team_name]["minutes_dominant_away"]))
    
    resp["avg_total_dominance"] = np.mean(list(last_games[last_games["team_home"] == team_name]["total_dominance_home"]) +  list(last_games[last_games["team_away"] == team_name]["total_dominance_away"]))
        
    resp["max_dominance"] = np.mean(list(last_games[last_games["team_home"] == team_name]["max_dominance_home"]) + list(last_games[last_games["team_away"] == team_name]["max_dominance_away"]))
    
    resp["min_dominance"] = np.mean(list(last_games[last_games["team_home"] == team_name]["min_dominance_home"]) + list(last_games[last_games["team_away"] == team_name]["min_dominance_away"]))
    
    resp["avg_dominance"] = np.mean(list(last_games[last_games["team_home"] == team_name]["max_dominance_home"]) + list(last_games[last_games["team_away"] == team_name]["max_dominance_away"]))
    
    return(pd.DataFrame(resp))

def prepara_base(base):
    """
        Faz a prepração inicial da base cru que vem do NBA_Stats. 
        Cria as variáveis básicas e junta o time mandante e o visitante na mesma linha.
    """
    base["fl_home"] = np.where(base["game"].str[6:9] == base["TEAM_ABBREVIATION"], 1, 0)
    
    home_games = base[base["fl_home"] == 1].set_index("GAME")
    away_games = base[base["fl_home"] == 0].set_index("GAME")
    
    all_games = home_games.join(away_games, how="inner", lsuffix="_home", rsuffix="_away")
    all_games.drop(["GAME_ID_away", "GAME_DATE_away", "GAME_PLACE_away",
                "MIN_home", "MIN_away", 'PTS_hustle_home', 'PTS_hustle_away',
                "fl_home_away", "fl_home_home"], axis=1, inplace=True, errors="ignore")
    
    all_games["data"] = [datetime.strptime(str(x), '%Y-%m-%d') for x in all_games.GAME_DATE_home]
    all_games["SEASON"] = [get_season(x) for x in all_games.DATE]
    all_games["fl_playoff"] = [is_playoff(x) for x in all_games.DATE]
    all_games['fl_home_win'] = np.where(all_games['PTS_home'] > all_games['PTS_away'], 1, 0)
    all_games = all_games.sort_values('DATE')
    
    all_games = all_games.rename(columns={'TEAM_ABBREVIATION_home': "team_home",
                                     'TEAM_ABBREVIATION_away': "team_away"})
    
    all_games["team_home_game_num"] = all_games.groupby(['team_home']).cumcount() + 1
    all_games["team_away_game_num"] = all_games.groupby(['team_away']).cumcount() + 1
    
    return(all_games)

def cria_features(new_games, all_games = None, dist_matrix_path="../old_files/dist_matrix_km.csv"):
    """
        Cria as features relacionadas à distância e fadiga, utilizando a localização e a data dos jogos ocorridos
        
        Parâmetros:
            new_games: DataFrame com os jogos a que se deseja preencher com as variáveis
            all_games: DataFrame com os jogos históricos ocorridos. Caso, None, assume-se que a base 'new_games' possua o histórico também.
     dist_matrix_path: Caminho do arquivo com tabela da matriz de distâncias entre os times
    """
    
    if(all_games is None):
        all_games = new_games.copy()
    
    df_dist = pd.read_csv(dist_matrix_path, index_col=0)
    
    new_games["DISTANCE_KM_home"] = [get_dist_last_game(all_games, x.DATE, df_dist, x.team_home, x.team_away, is_home=True) 
                                    for _, x in new_games.iterrows()]
    new_games["DISTANCE_KM_away"] = [get_dist_last_game(all_games, x.DATE, df_dist, x.team_home, x.team_away, is_home=False) 
                                    for _, x in new_games.iterrows()]
    
    new_games["DAYS_FROM_LAST_GAME_home"] = [get_days_from_last_game(all_games, x.DATE, x.team_home) 
                                            for _, x in new_games.iterrows()]
    new_games["DAYS_FROM_LAST_GAME_away"] = [get_days_from_last_game(all_games, x.DATE, x.team_away) 
                                            for _, x in new_games.iterrows()]
    
    return(new_games)
    
    
def gera_last_N_games(new_games, all_games = None, N = [5],
                      to_drop=["fl_win", "Total_passes", "result", "Accurate passes", "hora", "game"]):
    """
        Gera as variáveis em relação ao desempenho médio nos últimos N jogos nas visões LAST_GAMES, AS_HOME, AS_AWAY e RIVALS.
        
        Parâmetros:
            all_games: DataFrame com as informações do desempenho dos dois times por jogo. Tipicamente após aplicar as funções 'prepara_base()' e 'cria_features()'
            new_games: DataFrame com os novos jogos a serem computados os desempenhos passados. Deve conter as colunas 'team_home', 'team_away', 'DATE' e 'GAME'
                    N: Lista com os tamanhos de janelas dos últimos jogos a serem observados
    """
    
    gc.enable()
    resp = []
    
    if(all_games is None):
        all_games = new_games.copy()
    
    home_columns = [x for x in all_games.columns if x.endswith("_home") and x not in ['GAME_ID_home', 'TEAM_CITY_home', 'GAME_DATE_home', 'GAME_PLACE_home', 'TEAM_NICKNAME_home']]
    away_columns = [x for x in all_games.columns if x.endswith("_away") and x not in ['TEAM_CITY_away', 'TEAM_NICKNAME_away']]

    for index, row in tqdm_notebook(new_games.reset_index().iterrows()):
        game_line_n = []
        for n_games in N:
            # Home team
            home_last_games = get_last_games(all_games, row["data"], row["team_home"], n=n_games)
            home_last_games_as_home = get_last_games(all_games, row["data"], row["team_home"], filter="home", n=n_games)

            home_avg_last_games = get_avg_last_games(home_last_games, row["team_home"], home_columns, away_columns, data_ref=row["data"], to_drop=to_drop)
            home_avg_last_games_as_home = get_avg_last_games(home_last_games_as_home, row["team_home"], home_columns, away_columns, data_ref=row["data"], to_drop=to_drop)
            home_rivals_last_games = get_avg_last_games(home_last_games, row["team_home"], home_columns, away_columns, rivals=True, data_ref=row["data"], to_drop=to_drop)

            home_avg_last_games["game_ref"] = [row["game"]]
            home_avg_last_games.set_index("game_ref", inplace=True)
            home_avg_last_games.drop(["team_home", "team_away", 0] + to_drop,axis=1 ,errors="ignore", inplace=True)

            home_avg_last_games_as_home["game_ref"] = [row["game"]]
            home_avg_last_games_as_home.set_index("game_ref", inplace=True)
            home_avg_last_games_as_home.drop(["team_home", "team_away", 0], axis=1 ,errors="ignore", inplace=True)

            home_rivals_last_games["game_ref"] = [row["game"]]
            home_rivals_last_games.set_index("game_ref", inplace=True)
            home_rivals_last_games.drop(["team_home", "team_away", 0],axis=1 ,errors="ignore", inplace=True)

            #print(home_rivals_last_games.index, home_avg_last_games.index)

            # Away team
            away_last_games = get_last_games(all_games, row["data"], row["team_away"], n=n_games).reset_index()
            away_last_games_as_away = get_last_games(all_games, row["data"], row["team_away"], filter="away", n=n_games).reset_index()

            away_avg_last_games = get_avg_last_games(away_last_games, row["team_away"], home_columns, away_columns, data_ref=row["data"])
            away_avg_last_games_as_away = get_avg_last_games(away_last_games_as_away, row["team_away"], home_columns, away_columns, data_ref=row["data"])
            away_rivals_last_games = get_avg_last_games(away_last_games, row["team_away"], home_columns, away_columns, rivals=True, data_ref=row["data"])

            away_avg_last_games["game_ref"] = [row["game"]]
            away_avg_last_games.set_index("game_ref", inplace=True)
            away_avg_last_games.drop(["team_home", "team_away", 0] + to_drop,axis=1 ,errors="ignore", inplace=True)

            away_avg_last_games_as_away["game_ref"] = [row["game"]]
            away_avg_last_games_as_away.set_index("game_ref", inplace=True)
            away_avg_last_games_as_away.drop(["team_home", "team_away", 0] + to_drop,axis=1 ,errors="ignore", inplace=True)

            away_rivals_last_games["game_ref"] = [row["game"]]
            away_rivals_last_games.set_index("game_ref", inplace=True)
            away_rivals_last_games.drop(["team_home", "team_away", 0] + to_drop,axis=1 ,errors="ignore", inplace=True)

            #print(away_rivals_last_games.index, away_avg_last_games.index)

            # Junta bases 

            if (n_games == 10000):
                n_games_str = "ALL"
            else:
                n_games_str = str(n_games)

            avg_last_games = home_avg_last_games.join(away_avg_last_games, how="inner", 
                                 lsuffix='_home_L' + n_games_str, rsuffix='_away_L' + n_games_str).drop('level_0', axis=1, errors="ignore")

            avg_last_games_as = home_avg_last_games_as_home.join(away_avg_last_games_as_away, how="inner", 
                                     lsuffix='_home_L' + n_games_str + '_HOME', rsuffix='_away_L' + n_games_str + '_AWAY').drop('level_0', axis=1, errors="ignore")


            rivals_last_games = home_rivals_last_games.join(away_rivals_last_games, how="inner",
                                    lsuffix='_home_L' + n_games_str, rsuffix='_away_L' + n_games_str).drop('level_0', axis=1, errors="ignore")


            #print(rivals_last_games.columns)

            game_line = avg_last_games.join(rivals_last_games, how="inner", rsuffix="_rivals")

            game_line = game_line.join(avg_last_games_as, how="inner")

            game_line = pd.concat([row.to_frame().transpose().set_index("game"), game_line], axis=1)

            print(str(row["game"]) + " " + str(n_games), end="\r")

            game_line_n.append(game_line)

        resp.append(game_line_n)

        del home_last_games
        del home_last_games_as_home
        del home_avg_last_games
        del home_avg_last_games_as_home
        del home_rivals_last_games

        del away_last_games
        del away_last_games_as_away
        del away_avg_last_games
        del away_avg_last_games_as_away
        del away_rivals_last_games

        del avg_last_games
        del avg_last_games_as
        del game_line
        del game_line_n

        gc.collect()
        
    resp2 = []
    for r in resp:
        resp2.append(pd.concat(r, axis=1))
        
    del resp
    gc.collect()
    gc.disable()
    
    return(pd.concat(resp2))

def variaveis_delta(df_resp, N = [5], to_predict = True, keep_features = ["team_home", "team_away", "DATE",  
                                                     'DISTANCE_KM_home', 'DISTANCE_KM_away', 'DAYS_FROM_LAST_GAME_home',
                                                       'DAYS_FROM_LAST_GAME_away']):
    """
        Crias variáveis de Delta e Cross a partir da base com as features dos últimos N jogos do método 'gera_last_N_games()'.
            D1: (HOME - AWAY) -> Desempenho do time da casa subtraído do desempenho do time visitante
            D2: (AS_HOME - AS_AWAY) -> Desempenho do time da casa nos últimos jogos home subtraído do desempenho do time visitante nos últimos jogos away
            C1: (OPP_HOME - AWAY) -> Desempenho dos times oponentes do time da casa nos últimos jogos subtraído do desempenho do visitante nos últimos jogos
            C2: (OPP_AWAY - HOME) -> Desempenho dos times oponentes do time visitante nos últimos jogos subtraído do desempenho do mandante nos últimos jogos
        
        Parâmetros:
            df_resp: DataFrame dos jogos com as variáveis referente aos últimos jogos
                  N: Lista dos tamanhos das janelas a se criar as features de delta
      keep_features: Lista de features individuais dos jogos a serem mantidas na base resultante
    """
    columns_subtract = []
    for var in df_resp.columns:
        if "_home_L5" in var:
            columns_subtract.append(var.replace("_home_L5_HOME", "").replace("_opponent_home_L5", "").replace("_home_L5", ""))
    columns_subtract = list(set(columns_subtract))
    
    if(not to_predict):
        keep_features += ['fl_home_win', 'team_home_game_num', 'team_away_game_num']

    filtrada = df_resp[keep_features].copy()
    
    for column in columns_subtract:
        for n_games in N:            
            if (n_games == 10000):
                n_games_str = "ALL"
            else:
                n_games_str = str(n_games)
                
            print(n_games_str + " " + str(column) + "                        ", end="\r")

            filtrada["D1_" + column + "_L" + n_games_str] = df_resp[column + "_home_L" + n_games_str] - df_resp[column + "_away_L" + n_games_str]
            filtrada["D2_" + column + "_L" + n_games_str] = df_resp[column + "_home_L" + n_games_str + "_HOME"] - df_resp[column + "_away_L" + n_games_str + "_AWAY"]
            try:
                filtrada["C1_" + column + "_L" + n_games_str] = df_resp[column + "_opponent_home_L" + n_games_str] - df_resp[column + "_away_L" + n_games_str]
                filtrada["C2_" + column + "_L" + n_games_str] = df_resp[column + "_opponent_away_L" + n_games_str] - df_resp[column + "_home_L" + n_games_str]
            except KeyError:
                pass
            
    return(filtrada)

