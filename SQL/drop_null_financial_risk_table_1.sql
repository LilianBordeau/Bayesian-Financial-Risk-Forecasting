CREATE MULTISET TABLE DB_DATALAB_DAF.ML_FINANCIAL_RISK_2 AS 
(
SELECT *
FROM DB_DATALAB_DAF.ML_FINANCIAL_RISK
WHERE IDNT_COMP_FACT IS NOT NULL
) WITH DATA UNIQUE PRIMARY INDEX(ORDER_PACKAGE_NUMBER);