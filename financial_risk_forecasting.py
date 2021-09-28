import teradatasql
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import numpy as np

# Connexion Teradata
host     = '10.43.67.32'
user     = 'u165983'
password = '7BB43ryXd6'
url      = '{"host":"'+host+'","user":"'+user+'","password":"'+password+'"}'

# Connexion
connexion = teradatasql.connect(url)
curseur   = connexion.cursor()

# Paramètres
nb_factures = 12 # Nombre de factures sur lesquelles sera calculé le score de risque
date_min    = '2021-09-01' # Date de début pour la prédiction
date_max    = 'CURRENT_DATE' # Date de fin pour la prédiction

# Mapping variables
Y   = ['mtt_rk_'+str(i).zfill(2) for i in range(1,nb_factures)]
Y_n = [y+'_n' for y in Y] # Normalisé
Y_b = [y+'_b' for y in Y] # Binaire
Y_s = [y+'_s' for y in Y] # Scaled
Y_m = [y+'_m' for y in Y] # Minmax scaled
Y_r = [y+'_r' for y in Y] # Regression
Y_c = [y+'_c' for y in Y] # Classification
X_n = [x+'_n' for x in X] # Normalisé

def get_bayesian_tables_names():
    """ Récupère les nomes des tables bayesiennes.
    """

    table_bayes = pd.read_sql("""SELECT TableName 
                   FROM dbc.tablesv 
                   WHERE TableName LIKE '%ML_RFB_%'
                   AND CreatorName = '"""+user+"""'
                   ORDER BY CreateTimeStamp""", connexion).TableName.to_list()
    
    return table_bayes

def get_X(df_columns, table_bayes, seuil_cardinalite):
    """ Récupère les variables prédictives et filtre sur les cardinalités.
    """
    cardinalities = dict()
    
    with connexion.cursor() as cur:
        for table in table_bayes:
            cur.execute("""SELECT COUNT(*) FROM  DB_DATALAB_DAF."""+table+""";""")
            count = cur.fetchone()[0]
            connexion.commit()
            x = table.split('ML_RFB_')[1] 
            cardinalities[x] = count
            #print(x, count)
    
    # Filtre les variables prédictives avec une cardinalité trop grande pour empêcher le surapprentissage
    X = [k for k,v in cardinalities.items() if v < seuil_cardinalite]
    
    X = [x for x in X if x in df_columns]
    print(len(cardinalities),'-->',len(X))
            
    return X, cardinalities

def load_scalers():
    """ Charge les scalers utilisés lors de l'entrianement des modeles.
    """
    pickle_path = 'Scalers/scalers.pickle'
    with open(pickle_path,"rb") as f:
        scalers = pickle.load(f)
    return scalers

def wrapper_inference(df, Y):
    """
    Wrapper pour l'inference.
    Retourne: la dataframe avec les prédictions
    """
    # Dataframe pour stocker les résultats bruts
    df_res     = pd.DataFrame()
    df_res[['ORDER_PACKAGE_NUMBER', 'IDNT_COMP_FACT']] = df[['ORDER_PACKAGE_NUMBER','IDNT_COMP_FACT']]

    for y in Y:
        df_res[y] = df[y]

        # Reconstruit les tables encodées bayesiennes
        df_bay_reg, df_bay_cla = get_bayesian_tables(X,y)

        # Prédiction
        df_res = get_bayesian_predictions(df, df_bay_reg, df_bay_cla, df_metrics, df_res, X, y)
    
    return df_res

def load_table_inference_data(user):
    """
    Charge la table des données entre date_min et date_max pour l'inférence.
    """
    print('\tLoad dataset...')

    df = pd.read_sql("""SELECT ORDER_PACKAGE_NUMBER, 
                               IDNT_COMP_FACT 
                        FROM """+user+""".SUBSET_INFERENCE_DATA;""", connexion)

    df = df.loc[:,~df.columns.duplicated()]
    df = df.dropna(subset=Y)
    df[Y_b] = df[Y].applymap(lambda x: 0 if x >= 0 else 1)

    print('DF shape:', df.shape)
    return df

def create_table_inference_data(table_data, user, date_min, date_max):
    """
    Create la table des données entre date_min et date_max pour l'inférence.
    """

    query = """ CREATE MULTISET VOLATILE TABLE """+user+""".SUBSET_INFERENCE_DATA AS
            (
            SELECT T1.* FROM """+table_data+""" T1
            WHERE CAST(T1.RECORDED_DATE AS DATE) BETWEEN '"""+date_min+"""' AND '"""+date_max+"""'
            ) WITH DATA UNIQUE PRIMARY INDEX(ORDER_PACKAGE_NUMBER) ON COMMIT PRESERVE ROWS;"""

    print('\tCreate dataset...')

    with connexion.cursor() as cur:
        cur.execute(query)
        connexion.commit()

def drop_table_inference_data(user):
    """
    Supprime la table des données pour l'inférence.
    """

    with connexion.cursor() as cur:
        cur.execute("DROP TABLE "+user+".SUBSET_INFERENCE_DATA")
        connexion.commit()

def get_bayesian_tables(X,y):
    """
    Construit une table bayesienne à partir de la variable à prédire (y) et des features (X)
    grâce aux tables construites sur fast_bayesian_tables.ipynb 
    """
    
    print('\tGet bayesian for...', y)

    # Créer 2 tables temporaires pour reconstruire les associations
    
    query_reg = """ CREATE TABLE DB_DATALAB_DAF.ML_SCORING_BAYES_TMP_REG (
                            ORDER_PACKAGE_NUMBER VARCHAR(255) CHARACTER SET LATIN NOT CASESPECIFIC,
                            """+','.join([x+' FLOAT' for x in X])+"""
                    ) UNIQUE PRIMARY INDEX(ORDER_PACKAGE_NUMBER);"""

    query_cla = """ CREATE TABLE DB_DATALAB_DAF.ML_SCORING_BAYES_TMP_CLA (
                            ORDER_PACKAGE_NUMBER VARCHAR(255) CHARACTER SET LATIN NOT CASESPECIFIC,
                            """+','.join([x+' FLOAT' for x in X])+"""
                    ) UNIQUE PRIMARY INDEX(ORDER_PACKAGE_NUMBER);"""

    # Insert l'identifiant pour faire la jointure et des champs vides pour stocker les valeurs
    
    query_ins_reg = """INSERT INTO DB_DATALAB_DAF.ML_SCORING_BAYES_TMP_REG
                       SELECT ORDER_PACKAGE_NUMBER,
                              """+','.join(['NULL AS '+x for x in X])+"""
                       FROM """+user+""".SUBSET_INFERENCE_DATA;"""

    query_ins_cla = """INSERT INTO DB_DATALAB_DAF.ML_SCORING_BAYES_TMP_CLA
                       SELECT ORDER_PACKAGE_NUMBER,
                              """+','.join(['NULL AS '+x for x in X])+"""
                       FROM """+user+""".SUBSET_INFERENCE_DATA;"""

    with connexion.cursor() as cur:
        cur.execute(query_reg)
        connexion.commit()
        cur.execute(query_cla)
        connexion.commit()
        cur.execute(query_ins_reg)
        connexion.commit()
        cur.execute(query_ins_cla)
        connexion.commit()
    
    print('\tW.r.t: ', end='')
    for i, x in enumerate(X):
        print(i, end=' ')

        query_upd_cla = """
                UPDATE S1
                FROM DB_DATALAB_DAF.ML_SCORING_BAYES_TMP_CLA S1, (
                SELECT T0.ORDER_PACKAGE_NUMBER, T1."""+y+"""_b as """+x+"""
                                FROM """+user+""".SUBSET_INFERENCE_DATA T0

                                LEFT JOIN DB_DATALAB_DAF.ML_RFB_"""+x+""" T1
                                ON T0."""+x+""" = T1."""+x+""") S2
                SET """+x+""" = S2."""+x+"""
                WHERE S1.ORDER_PACKAGE_NUMBER = S2.ORDER_PACKAGE_NUMBER;
                """

        query_upd_reg = """
                   UPDATE S1
                FROM DB_DATALAB_DAF.ML_SCORING_BAYES_TMP_REG S1, (
                SELECT T0.ORDER_PACKAGE_NUMBER, T1."""+y+""" as """+x+"""
                                FROM """+user+""".SUBSET_INFERENCE_DATA T0

                                LEFT JOIN DB_DATALAB_DAF.ML_RFB_"""+x+""" T1
                                ON T0."""+x+""" = T1."""+x+""") S2
                SET """+x+""" = S2."""+x+"""
                WHERE S1.ORDER_PACKAGE_NUMBER = S2.ORDER_PACKAGE_NUMBER;
                """

        with connexion.cursor() as cur:
            cur.execute(query_upd_cla)
            connexion.commit()
            cur.execute(query_upd_reg)
            connexion.commit()
            
    print()

    df_bay_reg = pd.read_sql("""SELECT * FROM  DB_DATALAB_DAF.ML_SCORING_BAYES_TMP_REG""", connexion)
    df_bay_cla = pd.read_sql("""SELECT * FROM  DB_DATALAB_DAF.ML_SCORING_BAYES_TMP_CLA""", connexion)

    with connexion.cursor() as cur:
        cur.execute("DROP TABLE DB_DATALAB_DAF.ML_SCORING_BAYES_TMP_CLA;")
        connexion.commit()
        cur.execute("DROP TABLE DB_DATALAB_DAF.ML_SCORING_BAYES_TMP_REG;")
        connexion.commit()
            
    return df_bay_reg, df_bay_cla

def get_bayesian_predictions(df, df_bay_reg, df_bay_cla, df_metrics, df_res, X, y):
    """
    Calcule les prédictions pour une table df avec ses tables encodées bayesiennes (df_bay_reg, df_bay_cla)
    Mets à jour les tables df_metrics et df_res avec:
    - Les métriques de scores pour df_metrics
    - Les résultats bruts pour df_res
    Renvoie les deux à la fin.
    
    Suppose : avoir des modèles pré-entrainés sauvegardés (load avec pickle)
    """
    
    print('\tPrediction for...', y)

    # verifie que df et les tables bayesiennes sont bien alignées
    df_bay_cla = df_bay_cla[df_bay_cla.ORDER_PACKAGE_NUMBER.isin(df.ORDER_PACKAGE_NUMBER)]
    df_bay_reg = df_bay_reg[df_bay_reg.ORDER_PACKAGE_NUMBER.isin(df.ORDER_PACKAGE_NUMBER)]

    df_bay_cla = df_bay_cla.fillna(0)
    df_bay_reg = df_bay_reg.fillna(0)

    df_bay_cla = df_bay_cla.sort_values(by='ORDER_PACKAGE_NUMBER')
    df_bay_reg = df_bay_reg.sort_values(by='ORDER_PACKAGE_NUMBER')
    df         = df.sort_values(by='ORDER_PACKAGE_NUMBER')

    df_bay_cla.index = df.index
    df_bay_reg.index = df.index

    ##################################### CLASSIFY

    #display(df_cla)

    scaler   = MinMaxScaler()
    df_bay_cla[X_n] = scaler.fit_transform(df_bay_cla[X])

    X_cla      = df_bay_cla[X_n]
    y_test_cla = df[y+'_b']

    pickle_path = 'Models/cla_'+y+'.pickle'
    with open(pickle_path,"rb") as f:
        model_c = pickle.load(f)

    ##################################### 

    y_pred_cla  = model_c.predict(X_cla)
    y_pred_cla  = pd.Series(y_pred_cla, index=y_test_cla.index)

    df_res[y+'_c'] = y_pred_cla

    ##################################### 

    cm = confusion_matrix(y_test_cla, y_pred_cla)

    df_metrics.loc['Positives' ,y] = y_test_cla.sum()
    df_metrics.loc['Negatives' ,y] = len(y_test_cla) - y_test_cla.sum()
    df_metrics.loc['TP' ,y]        = cm[1,1]
    df_metrics.loc['TN' ,y]        = cm[0,0]
    df_metrics.loc['FP' ,y]        = cm[0,1]
    df_metrics.loc['FN' ,y]        = cm[1,0]
    df_metrics.loc['f1_score' ,y]  = f1_score(y_test_cla, y_pred_cla)
    df_metrics.loc['precision',y]  = precision_score(y_test_cla, y_pred_cla)
    df_metrics.loc['recall'   ,y]  = recall_score(y_test_cla, y_pred_cla)
    #df_metrics.loc['accuracy',y]  = accuracy_score(y_test_cla, y_pred_cla)

    ##################################### REGRESS

    pickle_path = 'Models/reg_POS_'+y+'.pickle'
    with open(pickle_path,"rb") as f:
        model_pos = pickle.load(f)

    pickle_path = 'Models/reg_NEG_'+y+'.pickle'
    with open(pickle_path,"rb") as f:
        model_neg = pickle.load(f)

    scaler_X_pos = scalers[y]['POS']['X']
    scaler_X_neg = scalers[y]['NEG']['X']
    scaler_y_pos = scalers[y]['POS']['y']
    scaler_y_neg = scalers[y]['NEG']['y']

    #####################################

    index_pos = y_pred_cla[y_pred_cla == 0].index
    index_neg = y_pred_cla[y_pred_cla == 1].index

    X_reg_pos = df_bay_reg.loc[index_pos,X]
    X_reg_neg = df_bay_reg.loc[index_neg,X]

    X_reg_pos[X] = scaler_X_pos.transform(X_reg_pos[X])
    X_reg_neg[X] = scaler_X_neg.transform(X_reg_neg[X])

    y_reg_pos = df.loc[index_pos,y]
    y_reg_neg = df.loc[index_neg,y]

    y_pred_pos  = model_pos.predict(X_reg_pos)
    y_pred_neg  = model_neg.predict(X_reg_neg)

    y_pred_pos  = pd.DataFrame(y_pred_pos, index=index_pos)
    y_pred_neg  = pd.DataFrame(y_pred_neg, index=index_neg)

    y_pred_pos = scaler_y_pos.inverse_transform(y_pred_pos)
    y_pred_neg = scaler_y_neg.inverse_transform(y_pred_neg)

    y_reg_pos = np.log(abs(y_reg_pos) + 1)
    y_reg_neg = np.log(abs(y_reg_neg) + 1)

    y_pred_pos =  (np.exp(y_pred_pos) - 1)
    y_pred_neg = -(np.exp(y_pred_neg) - 1)

    df_res.loc[y_reg_pos.index,y+'_r'] = y_pred_pos
    df_res.loc[y_reg_neg.index,y+'_r'] = y_pred_neg

    df_res = df_res.round(2)
    
    return df_res

def insert_results(df_res):
    """
    Insert les résultats sur les tables ML_SCORING_FINANCIAL_RISK_CLA (Classification) 
    et ML_SCORING_FINANCIAL_RISK_REG (Régression).
    En passant par la table temporaire DB_DATALAB_DAF.UPLOAD_RESULTS car FastLoad ne peut pas charger des
    données sur une table qui n'est pas vide.
    """

    values = df_res[['ORDER_PACKAGE_NUMBER','IDNT_COMP_FACT'] + Y_r].values.tolist()

    insert = """{fn teradata_require_fastload}
                INSERT INTO DB_DATALAB_DAF.UPLOAD_RESULTS (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""

    with connexion.cursor() as cur:
            cur.execute("{fn teradata_nativesql}{fn teradata_autocommit_off}")
            connexion.commit()
            cur.execute(insert, values)
            connexion.commit()
            cur.execute("""INSERT INTO DB_DATALAB_DAF.ML_SCORING_FINANCIAL_RISK_REG 
                           SELECT * FROM DB_DATALAB_DAF.UPLOAD_RESULTS;""")
            connexion.commit()
            cur.execute("""DELETE DB_DATALAB_DAF.UPLOAD_RESULTS ALL;""")
            connexion.commit()

    values = df_res[['ORDER_PACKAGE_NUMBER','IDNT_COMP_FACT'] + Y_c].values.tolist()

    insert = """{fn teradata_require_fastload}
                INSERT INTO DB_DATALAB_DAF.UPLOAD_RESULTS (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""

    with connexion.cursor() as cur:
            cur.execute("{fn teradata_nativesql}{fn teradata_autocommit_off}")
            connexion.commit()
            cur.execute(insert, values)
            connexion.commit()
            cur.execute("""INSERT INTO DB_DATALAB_DAF.ML_SCORING_FINANCIAL_RISK_CLA 
                           SELECT * FROM DB_DATALAB_DAF.UPLOAD_RESULTS;""")
            connexion.commit()
            cur.execute("""DELETE DB_DATALAB_DAF.UPLOAD_RESULTS ALL;""")
            connexion.commit()

def create_table_results():
    
    create1 = open(repertoire + 'forecasting_create_risk_cla.sql', 'r').read()
    create2 = open(repertoire + 'forecasting_create_risk_reg.sql', 'r').read()
    create3 = open(repertoire + 'forecasting_create_upload_results.sql', 'r').read()

    with connexion.cursor() as cur:
        cur.execute(create1)
        connexion.commit()
        cur.execute(create2)
        connexion.commit()
        cur.execute(create3)
        connexion.commit()

if __name__ == "__main__":
    
    # Charge les données pour les dates spécifiées
    create_table_inference_data(table_data, user, date_min, date_max)
    df = load_table_inference_data(user)
    
    # Inférence
    df_res = wrapper_inference(df, Y)
    
    # Insère les résultats
    create_table_results()
    insert_results(df_res)

    # Supprime la table des données
    drop_table_inference_data(user)