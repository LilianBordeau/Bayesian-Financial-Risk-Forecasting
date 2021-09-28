import teradatasql

# Connexion Teradata
host     = '10.43.67.32'
user     = 'u165983'
password = '7BB43ryXd6'
url      = '{"host":"'+host+'","user":"'+user+'","password":"'+password+'"}'

# Connexion
connexion = teradatasql.connect(url)
curseur   = connexion.cursor()

# Paramètres
nb_factures   = 12 # Nombre de factures sur lesquelles sera calculé le score de risque
date_min      = '2021-06-01' # Date de départ du jeu de données
date_max      = 'CURRENT_DATE' # Date de fin du jeu de données
table_data    = 'VH_S_BANCAIRE.TDI_ECOM_CMD' # Table des données de commandes
table_bills_m = 'VH_B_FINANCE.TDI_ARBOR_SOLDE_FACTURE' # Table des factures mobiles
table_bills_f = 'VH_B_FINANCE.'

# Répertoire SQL
repertoire = r'SQL/'

# X et Y
to_exclude = ['ORDER_PACKAGE_NUMBER', 'TITULAIRE_CLARIFY_ID']

def get_categorical_X(table_data, to_exclude=[]):
    """ Récupère les variables catégoriques de la table de données.
    """

    X = list()

    with connexion.cursor() as cur:
        cur.execute("""SHOW TABLE """+table_data+""";""")
        table_spec = cur.fetchone()[0].replace('\r', '\n')

        for line in table_spec.splitlines():
            if 'VARCHAR' in line:
                col = line.split('VARCHAR')[0].strip()
                if col not in to_exclude:
                    X.append(col)
        connexion.commit()
    
    return X

def get_cardinalities(X):
    cardinalities = dict()
    
    with connexion.cursor() as cur:
        for x in X:
            cur.execute("""SELECT COUNT(*) FROM  DB_DATALAB_DAF.ML_RFB_"""+X+""";""")
            count = cur.fetchone()[0]
            connexion.commit()
            cardinalities[x] = count
            print(x, count)
            
    return cardinalities

def create_financial_risk_table(date_min, date_max, table_data):
    """ Créer une table vide avec correspondance ID_COMMANDE <> COMPTE_FACTURATION du jeu de données initial.
    """

    # initialise table avec données facturation fixe
    query1 = open(repertoire + 'create_financial_risk_table.sql', 'r').read()
    query1 = query1.replace('TABLE_DATA', table_data)
    query1 = query1.replace('DATE_MIN', date_min)
    query1 = query1.replace('DATE_MAX', date_max)
    
    # met à jour avec données facturation mobile
    query2 = open(repertoire + 'update_financial_risk_table.sql', 'r').read()
        
    # supprime les valeurs nulles
    query3 = open(repertoire + 'drop_null_financial_risk_table_1.sql', 'r').read()
    query4 = open(repertoire + 'drop_null_financial_risk_table_2.sql', 'r').read()
    query5 = open(repertoire + 'drop_null_financial_risk_table_3.sql', 'r').read()

    # création de la table
    with connexion.cursor() as cur:
        cur.execute(query1)
        cur.execute(query2)
        cur.execute(query3)
        cur.execute(query4)
        cur.execute(query5)

def create_financial_bills_table(nb_factures, user_login):
    """ Récupère les factures nécessaires au calcul du risque financier.
    """

    # table avec données facturation mobile
    query1 = open(repertoire + 'create_financial_bills_mob.sql', 'r').read()
    query1 = query1.replace('USER_LOGIN', user_login)
    query1 = query1.replace('NB_FACTURES', nb_factures)
    
    # table avec données facturation fixe
    query2 = open(repertoire + 'create_financial_bills_fix.sql', 'r').read()
    query2 = query2.replace('USER_LOGIN', user_login)
    query2 = query2.replace('NB_FACTURES', nb_factures)

    # création de la table
    with connexion.cursor() as cur:
        cur.execute(query1)
        cur.execute(query2)

def populate_financial_risk_table(nb_factures, user_login):
    """ Ajoute les valeurs de risque cumulé à ML_FINANCIAL_RISK.
    """
    
    for i in range(1,nb_factures):
        mtt_col = 'mtt_rk_'+str(i).zfill(2)
        print(mtt_col)
        
        query_add_col = """ALTER TABLE DB_DATALAB_DAF.ML_SCORING_FINANCIAL_RISK ADD """+mtt_col+""" FLOAT;"""
        
        # update avec données facturation mobile
        query1 = open(repertoire + 'populate_financial_risk_table.sql', 'r').read()
        query1 = query1.replace('USER_LOGIN', user_login)
        query1 = query1.replace('MTT_COL', mtt_col)
        query1 = query1.replace('NB_FACTURES', i)
        query1 = query1.replace('TABLE_BILLS', 'ML_FINANCIAL_BILLS_MOB')
        
        # update avec données facturation fixe
        query2 = open(repertoire + 'populate_financial_risk_table.sql', 'r').read()
        query2 = query2.replace('USER_LOGIN', user_login)
        query2 = query2.replace('MTT_COL', mtt_col)
        query2 = query2.replace('NB_FACTURES', i)
        query2 = query2.replace('TABLE_BILLS', 'ML_FINANCIAL_BILLS_MOB')
        
        # création de la table
        with connexion.cursor() as cur:
            cur.execute(query1)
            cur.execute(query2)

def wrapper_financial_risk_table(nb_factures, user_login, date_min, date_max, table_data):
    """ Wrapper pour créer le risque financier sur tous les comptes du jeu de données.
    """
    
    create_financial_risk_table(date_min, date_max, table_data)
    create_financial_bills_table(nb_factures, user_login)
    populate_financial_risk_table(nb_factures, user_login)

def create_bayesian_tables(X, table_data):
    """ Créer une table vide pour chaque variable prédictive du jeu de donneés.
    """

    for x in X:
        print(x)

        query = """CREATE TABLE DB_DATALAB_DAF.ML_RFB_"""+x+""" AS 
                   (
                        SELECT DISTINCT """+x+""" FROM """+table_data+"""
                   ) WITH DATA UNIQUE PRIMARY INDEX ("""+x+""");"""
        
        with connexion.cursor() as cur:
            cur.execute(query)
            connexion.commit()

def encode_continuous_variables(X, Y, table_data):
    """ Encode les variables X en fonction des variables Y continues.
    
    # create table BAYES_OUTCOME_NAME with columns = features, index = order_package_number
    # for feature in features:
    #    create volatile table with columns feature, frequency
    #    frequency <- mean(outcome) w.r.t. feature
    #    update join volatile table x table BAYES_OUTCOME_NAME on feature
    """

    for x in X:
        print(x)

        for y in Y:
            print(y, end=' ')

            query_add_col = """ALTER TABLE DB_DATALAB_DAF.ML_RFB_"""+x+""" ADD """+y+""" FLOAT;"""

            query_update  = """
                UPDATE T1
                FROM DB_DATALAB_DAF.ML_RFB_"""+x+""" T1, 
                (
                    SELECT T1."""+x+""",
                    AVG(ZEROIFNULL("""+y+""")) AS """+y+"""

                    FROM """+table_data+""" T1
                    LEFT JOIN DB_DATALAB_DAF.ML_SCORING_FINANCIAL_RISK T2
                    ON T1.IDNT_COMP_FACT = T2.IDNT_COMP_FACT
                    group by 1
                ) T2
                SET """+y+""" = T2."""+y+"""
                WHERE T1."""+x+""" = T2."""+x+""";"""

            with connexion.cursor() as cur:
                cur.execute(query_add_col)
                connexion.commit()
                cur.execute(query_update)
                connexion.commit()

        print('\n')

def encode_categorical_variables(X, Y, table_data):
    """ Encode les variables X en fonction des variables Y catégoriques.
    
    # create table BAYES_OUTCOME_NAME with columns = features, index = order_package_number
    # for feature in features:
    #    create volatile table with columns feature, frequency
    #    frequency <- mean(outcome) w.r.t. feature
    #    update join volatile table x table BAYES_OUTCOME_NAME on feature
    """
    
    for x in X:
        print(x,':', end=' ')

        for y in Y:
            print(y, end=' ')

            query_add_col = """ALTER TABLE DB_DATALAB_DAF.ML_RFB_"""+x+""" ADD """+y+"""_B FLOAT;"""

            query_update  = """
                UPDATE T1
                FROM DB_DATALAB_DAF.ML_RFB_"""+x+""" T1, 
                (
                    SELECT T1."""+x+""",
                    CAST(SUM((CASE WHEN """+y+""" >= 0 THEN 0 ELSE 1 END))  AS FLOAT) / CAST(COUNT(*) AS FLOAT) AS """+y+"""_B

                    FROM """+table_data+""" T1
                    LEFT JOIN DB_DATALAB_DAF.ML_SCORING_FINANCIAL_RISK T2
                    ON T1.IDNT_COMP_FACT = T2.IDNT_COMP_FACT
                    group by 1
                ) T2
                SET """+y+"""_B = T2."""+y+"""_B
                WHERE T1."""+x+""" = T2."""+x+""";"""

            with connexion.cursor() as cur:
                cur.execute(query_add_col)
                connexion.commit()
                cur.execute(query_update)
                connexion.commit()

        print()

def wrapper_bayesian_tables(X, Y, table_data):
    """ Wrapper pour la création des tables encodées bayesiennes.
    """
    
    create_bayesian_tables(X, table_data)
    encode_continuous_variables(X, Y, table_data)
    encode_categorical_variables(X, Y, table_data)

if __name__ == "__main__":
    
    X = get_categorical_X(table_data, to_exclude)
    Y = ['mtt_rk_'+str(i).zfill(2) for i in range(1,nb_factures)]
    
    wrapper_financial_risk_table(nb_factures, user_login, date_min, date_max, table_data)
    wrapper_bayesian_tables(X, Y, table_data)