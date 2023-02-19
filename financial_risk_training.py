import teradatasql
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score
import numpy as np

# Connexion Teradata
host     = ''
user     = ''
password = ''
url      = '{"host":"'+host+'","user":"'+user+'","password":"'+password+'"}'

# Connexion
connexion = teradatasql.connect(url)
curseur   = connexion.cursor()

# Paramètres
nb_factures   = 12 # Nombre de factures sur lesquelles sera calculé le score de risque
Y = ['mtt_rk_'+str(i).zfill(2) for i in range(1,nb_factures)] # nom des colonnes réponses
seuil_cardinalite = 1000000 # seuil de cardinalité pour filtrer les tables bayesiennes
n_sample = 10000 # nombre d'échantillons pour entrainer

# Mapping données
table_data = 'ML_SCORING_DATA'
table_risk = 'ML_SCORING_FINANCIAL_RISK'

# XGBOOST PARAMETERS Paramètres pour les modèles de régression
parameters_reg = {
            'n_estimators':500, #500
            'reg_lambda':1,
            'gamma':0.5, #0.5
            'eta':0.1, #0.1
            'max_depth':20,#20
            'objective':'reg:squarederror',
            "tree_method": "gpu_hist",
            'n_jobs': -1
            }

# XGBOOST PARAMETERS Paramètres pour les modèles de classification
parameters_cla = {
            'n_estimators':250, #500
            'reg_lambda':1,
            'gamma':0.5, #0.5
            'eta':0.1, #0.1
            'max_depth':20, #20
            'objective':'binary:logistic',
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            "tree_method": "gpu_hist",
            'n_jobs': -1
        }

def create_ramdom_sample_data(table_data, n_sample):
    """ Créer une table avec un échantillon aléatoire du jeu de données.
    """
    
    query = """ CREATE VOLATILE TABLE """+user+""".SUBSET_TRAINING_DATA_ALL AS
                (
                    SELECT T1.*, """+','.join(Y)+"""
                    FROM """+table_data+""" T1
                    INNER JOIN """+table_risk+""" T3
                    ON T1.IDNT_COMP_FACT = T3.IDNT_COMP_FACT
                    SAMPLE """+str(n_sample)+"""
                ) WITH DATA UNIQUE PRIMARY INDEX(ORDER_PACKAGE_NUMBER) ON COMMIT PRESERVE ROWS;"""

    with connexion.cursor() as cur:
        cur.execute(query)
        connexion.commit()

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
                       FROM """+user+""".SUBSET_TRAINING_DATA_ALL;"""

    query_ins_cla = """INSERT INTO DB_DATALAB_DAF.ML_SCORING_BAYES_TMP_CLA
                       SELECT ORDER_PACKAGE_NUMBER,
                              """+','.join(['NULL AS '+x for x in X])+"""
                       FROM """+user+""".SUBSET_TRAINING_DATA_ALL;"""

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
                                FROM """+user+""".SUBSET_TRAINING_DATA_ALL T0

                                LEFT JOIN DB_DATALAB_DAF.ML_RFB_"""+x+""" T1
                                ON T0."""+x+""" = T1."""+x+""") S2
                SET """+x+""" = S2."""+x+"""
                WHERE S1.ORDER_PACKAGE_NUMBER = S2.ORDER_PACKAGE_NUMBER;
                """

        query_upd_reg = """
                UPDATE S1
                FROM DB_DATALAB_DAF.ML_SCORING_BAYES_TMP_REG S1, (
                SELECT T0.ORDER_PACKAGE_NUMBER, T1."""+y+""" as """+x+"""
                                FROM """+user+""".SUBSET_TRAINING_DATA_ALL T0

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

def preprocessing_classifiers(df, df_bayes, X):
    """ Traitement en amont pour les classifieurs.
    """
    
    df_bayes = df_bayes[df_bayes.ORDER_PACKAGE_NUMBER.isin(df.ORDER_PACKAGE_NUMBER)]
    df_bayes = df_bayes.fillna(0)
    df = df.sort_values(by='ORDER_PACKAGE_NUMBER')
    df_bayes = df_bayes.sort_values(by='ORDER_PACKAGE_NUMBER')
    scaler   = MinMaxScaler()
    df_bayes[X_n] = scaler.fit_transform(df_bayes[X])
    return df_bayes

def preprocessing_df(df):
    """ Traitement en amont pour les données en général.
    """
    
    df_res = df.copy()
    
    # scale et filtre outlier
    scaler = RobustScaler()
    df_res[Y_b] = df_res[Y].applymap(lambda x: 0 if x >= 0 else 1)
    df_res[Y_n] = scaler.fit_transform(df_res[Y])
    df_res[Y_n] = df_res[Y_n][(-5 < df_res[Y_n]) & (df_res[Y_n] < 5)]

    # drop outliers
    df_res = df_res.dropna(subset=Y_n)

    # scale y bis
    scaler_min_max = MinMaxScaler()
    df_res[Y_m] = scaler_min_max.fit_transform(df_res[Y_n])
    
    return df_res

def wrapper_get_bayesian_tables(X,Y):
    """
    Wrapper qui renvoie toutes les tables bayésiennes pour un ensemble de variables Y, 
    et encodées categorique ou continues.
    """

    dfs_bayesian_cla = dict()
    dfs_bayesian_reg = dict()

    for y in Y:
        df_bay_reg, df_bay_cla = get_bayesian_tables(X,y)
        dfs_bayesian_cla[y] = df_bay_cla
        dfs_bayesian_reg[y] = df_bay_reg
        
    return dfs_bayesian_cla, dfs_bayesian_reg

def train_classifier():
    """
    Entraîne les XGBoost classifieurs.
    Sauvegarde les modèles sous Models/cla_*
    Sauvegarde les métriques de performances sous Results_Training/Classifiers.csv
    """
    df_metrics = pd.DataFrame()

    for y in Y:
        print(y)

        df_bayes = preprocessing_classifiers(df, dfs_bayesian_cla[y], X)

        X_cla = df_bayes[X]
        y_cla = df[y+'_b']

        X_train_cla, X_test_cla, y_train_cla, y_test_cla = train_test_split(X_cla, y_cla, test_size=0.33, random_state=42)

        parameters_cla['scale_pos_weight'] = (len(y_train_cla) - y_train_cla.sum()) / y_train_cla.sum()

        model_c = XGBClassifier(**parameters_cla)

        model_c.fit(X_train_cla, y_train_cla)

        pickle_path = 'Models/cla_'+y+'.pickle'
        pickle.dump(model_c, open(pickle_path, 'wb'))

        y_pred_cla  = model_c.predict(X_test_cla)
        y_pred_cla  = pd.Series(y_pred_cla, index=y_test_cla.index)

        cm = confusion_matrix(y_test_cla, y_pred_cla)

        df_metrics.loc['Positives', y] = y_test_cla.sum()
        df_metrics.loc['Negatives', y] = len(y_test_cla) - y_test_cla.sum()
        df_metrics.loc['TP', y]        = cm[1,1]
        df_metrics.loc['TN', y]        = cm[0,0]
        df_metrics.loc['FP', y]        = cm[0,1]
        df_metrics.loc['FN', y]        = cm[1,0]
        df_metrics.loc['f1_score', y]  = f1_score(y_test_cla, y_pred_cla)
        df_metrics.loc['precision', y]  = precision_score(y_test_cla, y_pred_cla)
        df_metrics.loc['recall', y]  = recall_score(y_test_cla, y_pred_cla)

    df_metrics.astype(float).round(2).to_csv('Results_Training/Classifiers.csv', sep=';')

def train_regressors():
    """
    Entraîne les XGBoost régresseurs.
    Sauvegarde les modèles sous Models/reg_*
    Sauvegarde les métriques de performances sous Results_Training/Regressors.csv
    Et les 'scalers' pour la mise à l'échelle sous Scalers/scalers.pickle
    """

    df_metrics = pd.DataFrame()
    df_res = pd.DataFrame()

    scalers = dict()

    for y, subset in df_regressions.items():
        print(y)
        scalers[y] = dict()
        for signe, df_train in subset.items():
            print(signe)

            df_train = df_train.fillna(0)
            df_train[y] = np.log(df_train[y].abs() + 1)

            scaler_X   = RobustScaler()
            scaler_y   = RobustScaler()

            df_train[X] = scaler_X.fit_transform(df_train[X])
            df_train[y] = scaler_y.fit_transform(df_train[[y]])

            scalers[y][signe] = dict()
            scalers[y][signe]['X'] = scaler_X
            scalers[y][signe]['y'] = scaler_y

            df_train      = df_train.dropna(subset=X)

            X_reg = df_train[X]
            y_reg = df_train[y]

            X_train, X_test, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.33, random_state=42)

            model_r = XGBRegressor(**parameters_reg)
            model_r.fit(X_train, y_train_reg)

            pickle_path = 'Models/reg_'+signe+'_'+y+'.pickle'
            pickle.dump(model_r, open(pickle_path, 'wb'))

            y_pred_reg = model_r.predict(X_test)

            df_metrics.loc['MAE_'+signe,y] = mean_absolute_error(y_test_reg, y_pred_reg)
            df_metrics.loc['MDE_'+signe,y] = median_absolute_error(y_test_reg, y_pred_reg)
            df_metrics.loc['MSE_'+signe,y] = mean_squared_error(y_test_reg, y_pred_reg)
            df_metrics.loc['RMS_'+signe,y] = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
            df_metrics.loc['R2S_'+signe,y] = r2_score(y_test_reg, y_pred_reg)

    pickle_path = 'Scalers/scalers.pickle'
    pickle.dump(scalers, open(pickle_path, 'wb'))
    df_metrics.astype(float).round(2).to_csv('Results_Training/Regressors.csv', sep=';')

def get_balanced_signed_sample():
    """
    Retourne un ensemble de jeux équilibrés (en terme montant positif/négatif)
    Pour entrainer les régresseurs dessus.
    """

    df_regressions = dict()

    for y in Y:
        print(y)

        sample = 2500


        query = """ CREATE VOLATILE TABLE """+user+""".SUBSET_TRAINING_DATA_POS_OR_NEG AS
                    (
                    SELECT T1.*, T3."""+y+"""

                    FROM DB_DATALAB_DAF.ML_SCORING_DATA T1

                    INNER JOIN DB_DATALAB_DAF.ML_SCORING_FINANCIAL_RISK T3
                    ON T1.IDNT_COMP_FACT = T3.IDNT_COMP_FACT

                    CONDITION_CLAUSE

                    SAMPLE """+str(sample)+"""
                    ) WITH DATA UNIQUE PRIMARY INDEX(ORDER_PACKAGE_NUMBER) ON COMMIT PRESERVE ROWS;"""

        query_pos = query.replace('POS_OR_NEG', 'POS')
        query_neg = query.replace('POS_OR_NEG', 'NEG')
        query_pos = query_pos.replace('CONDITION_CLAUSE', """WHERE T3."""+y+""" >= 0""")
        query_neg = query_neg.replace('CONDITION_CLAUSE', """WHERE T3."""+y+""" < 0""")

        with connexion.cursor() as cur:
            cur.execute(query_pos)
            connexion.commit()
            cur.execute(query_neg)
            connexion.commit()

        df_regressions[y] = dict()

        for signe in ['POS', 'NEG']:
            print(signe)

            df_tmp = pd.read_sql("""SELECT ORDER_PACKAGE_NUMBER, """+y+""" 
                                    FROM  """+user+""".SUBSET_TRAINING_DATA_"""+signe, connexion)

            # Reconstruct Bayesian Tables
            for x in X:
                #print(x,end=' ')

                query = """SELECT T0.ORDER_PACKAGE_NUMBER, T1."""+y+""" 
                           FROM """+user+""".SUBSET_TRAINING_DATA_"""+signe+""" T0
                           LEFT JOIN DB_DATALAB_DAF.ML_RFB_"""+x+""" T1 
                           ON T0."""+x+""" = T1."""+x

                df_tmp2         = pd.read_sql(query, connexion)
                df_tmp2.columns = ['ORDER_PACKAGE_NUMBER', x]
                df_tmp          = pd.merge(df_tmp, df_tmp2, on='ORDER_PACKAGE_NUMBER')

            with connexion.cursor() as cur:
                cur.execute("""DROP TABLE """+user+""".SUBSET_TRAINING_DATA_"""+signe+""";""")
                connexion.commit()

            df_regressions[y][signe] = df_tmp
    
    return df_regressions

def compute_complete_system_results():
    """
    Évalue le système dans son ensemble (classification puis régression selon les résultats de la classification).
    Sauvegarde les métriques de performances sous Results_Training/Complete_System.csv
    """
    
    df_metrics = pd.DataFrame()
    df_res     = pd.DataFrame()

    for y in Y:
        print(y)

        df_cla = preprocessing_classifiers(df, dfs_bayesian_cla[y], X)
        df_reg = preprocessing_classifiers(df, dfs_bayesian_reg[y], X)

        df     = df    .sort_values(by='ORDER_PACKAGE_NUMBER')
        df_cla = df_cla.sort_values(by='ORDER_PACKAGE_NUMBER')
        df_reg = df_reg.sort_values(by='ORDER_PACKAGE_NUMBER')

        df_cla.index = df.index
        df_reg.index = df.index

        print(df.shape, df_cla.shape, df_reg.shape)

        X_cla = df_cla[X]
        y_cla = df[y+'_b']

        X_train_cla, X_test_cla, y_train_cla, y_test_cla = train_test_split(X_cla, y_cla, test_size=0.33, random_state=42)

        ##################################### CLASSIFY

        pickle_path = 'Models/cla_'+y+'.pickle'
        with open(pickle_path,"rb") as f:
            model_c = pickle.load(f)

        y_pred_cla  = model_c.predict(X_test_cla)
        y_pred_cla  = pd.Series(y_pred_cla, index=y_test_cla.index)

        cm = confusion_matrix(y_test_cla, y_pred_cla)

        df_metrics.loc['Positives',y] = y_test_cla.sum()
        df_metrics.loc['Negatives',y] = len(y_test_cla) - y_test_cla.sum()
        df_metrics.loc['TP'       ,y] = cm[1,1]
        df_metrics.loc['TN'       ,y] = cm[0,0]
        df_metrics.loc['FP'       ,y] = cm[0,1]
        df_metrics.loc['FN'       ,y] = cm[1,0]
        df_metrics.loc['f1_score' ,y] = f1_score(y_test_cla, y_pred_cla)
        df_metrics.loc['precision',y] = precision_score(y_test_cla, y_pred_cla)
        df_metrics.loc['recall'   ,y] = recall_score(y_test_cla, y_pred_cla)

        ##################################### REGRESS

        y_pred_reg = model_r.predict(X_test)

        pickle_path = 'Models/reg_POS_'+y+'.pickle'
        with open(pickle_path,"rb") as f:
            model_pos = pickle.load(f)

        pickle_path = 'Models/reg_NEG_'+y+'.pickle'
        with open(pickle_path,"rb") as f:
            model_neg = pickle.load(f)

        df_reg = df_reg.fillna(0)

        scaler_X_pos = scalers[y]['POS']['X']
        scaler_X_neg = scalers[y]['NEG']['X']
        scaler_y_pos = scalers[y]['POS']['y']
        scaler_y_neg = scalers[y]['NEG']['y']

        y_reg_name = y

        index = y_pred_cla.index
        df_res[y] = df.loc[index,y]

        index_pos = y_pred_cla[y_pred_cla == 0].index
        index_neg = y_pred_cla[y_pred_cla == 1].index

        X_reg_pos = df_reg.loc[index_pos,X]
        X_reg_neg = df_reg.loc[index_neg,X]

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

        #y_reg_pos = np.log(abs(y_reg_pos) + 1)
        #y_reg_neg = np.log(abs(y_reg_neg) + 1)

        y_pred_pos =  (np.exp(y_pred_pos) - 1)
        y_pred_neg = -(np.exp(y_pred_neg) - 1)

        df_res.loc[y_reg_pos.index,y+'pred'] = y_pred_pos
        df_res.loc[y_reg_neg.index,y+'pred'] = y_pred_neg
        df_res.loc[y_reg_pos.index,y+'norm'] = y_reg_pos
        df_res.loc[y_reg_neg.index,y+'norm'] = y_reg_neg

        df_metrics.loc['POS mean_absolute_error'    ,y] = mean_absolute_error(y_reg_pos, y_pred_pos)
        df_metrics.loc['POS median_absolute_error'  ,y] = median_absolute_error(y_reg_pos, y_pred_pos)
        df_metrics.loc['POS mean_squared_error'     ,y] = mean_squared_error(y_reg_pos, y_pred_pos)
        df_metrics.loc['POS root_mean_squared_error',y] = np.sqrt(mean_squared_error(y_reg_pos, y_pred_pos))
        df_metrics.loc['POS r2_score'               ,y] = r2_score(y_reg_pos, y_pred_pos)

        df_metrics.loc['NEG mean_absolute_error'    ,y] = mean_absolute_error(y_reg_neg, y_pred_neg)
        df_metrics.loc['NEG median_absolute_error'  ,y] = median_absolute_error(y_reg_neg, y_pred_neg)
        df_metrics.loc['NEG mean_squared_error'     ,y] = mean_squared_error(y_reg_neg, y_pred_neg)
        df_metrics.loc['NEG root_mean_squared_error',y] = np.sqrt(mean_squared_error(y_reg_neg, y_pred_neg))
        df_metrics.loc['NEG r2_score'               ,y] = r2_score(y_reg_neg, y_pred_neg)

        sums_pos = df_res[df_res[y] >= 0].sum()
        sums_neg = df_res[df_res[y] <  0].sum()
        sums_tot = df_res[y+'norm'] - df_res[y+'pred']

        df_metrics.loc['% of negative amount' , y_reg_name] = sums_neg[y+'pred'] / sums_neg[y+'norm']
        df_metrics.loc['% of positive amount' , y_reg_name] = sums_pos[y+'pred'] / sums_pos[y+'norm']

        df_metrics.loc['% of negative amount' , y_reg_name] = sums_neg[y+'pred'] / sums_neg[y+'norm']
        df_metrics.loc['% of positive amount' , y_reg_name] = sums_pos[y+'pred'] / sums_pos[y+'norm']
        df_metrics.loc['% of total amount'    , y_reg_name] = ((df_res[y+'norm'] - df_res[y+'pred']).sum() / df_res[y+'norm'].sum()) + 1

    df_metrics.astype(float).round(2).to_csv('Results_Training/Complete_System.csv', sep=';')

if __name__ == "__main__":
    create_ramdom_sample_data(table_data, n_sample)

    df = pd.read_sql("""SELECT * FROM  """+user+""".SUBSET_TRAINING_DATA_ALL;""", connexion)
    df = df.loc[:,~df.columns.duplicated()]

    table_bayes = get_bayesian_tables_names()
    X, cardinalities = get_X(df.columns.str.lower(), table_bayes, seuil_cardinalite)

    Y_n = [y+'_n' for y in Y] # Normalisé
    Y_b = [y+'_b' for y in Y] # Binaire
    Y_s = [y+'_s' for y in Y] # Scaled
    Y_m = [y+'_m' for y in Y] # Min-Max Scaled
    X_n = [x+'_n' for x in X] # Normalisé

    df = preprocessing_df(df)
    dfs_bayesian_cla, dfs_bayesian_reg = wrapper_get_bayesian_tables(X,Y)
    df_regressions = get_balanced_signed_sample()

    train_classifier()
    train_regressors()
    compute_complete_system_results()