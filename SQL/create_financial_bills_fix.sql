CREATE VOLATILE MULTISET TABLE USER_LOGIN.ML_FINANCIAL_BILLS_FIX AS (
SELECT 
T1.IDNT_COMP_FACT, 
T2.BUDAT,
Row_Number() Over (PARTITION BY IDNT_COMP_FACT ORDER BY BUDAT ASC) AS rk,
Abs(Sum(MTT_TTC)) AS GAINS,
Abs(Sum(MTT_PNS)) AS PERTES,
(GAINS - PERTES) AS RISQUE
FROM DB_DATALAB_DAF.ML_FINANCIAL_RISK T1
INNER JOIN DB_DATALAB_DAF.PERM_1757_00 T2
ON T1.IDNT_COMP_FACT = T2.VKONT
GROUP BY 1, 2
QUALIFY rk < NB_FACTURES
) WITH DATA UNIQUE PRIMARY INDEX (IDNT_COMP_FACT, bill_ref_no);