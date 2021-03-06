UPDATE T1
FROM DB_DATALAB_DAF.ML_SCORING_FINANCIAL_RISK T1, (
SELECT IDNT_COMP_FACT, (Sum(GAINS) - Sum(PERTES)) AS MTT_COL
FROM USER_LOGIN.TABLE_BILLS
WHERE RK <= NB_FACTURES
GROUP BY 1) T2
SET MTT_COL = T2.MTT_COL
WHERE T1.IDNT_COMP_FACT = T2.IDNT_COMP_FACT