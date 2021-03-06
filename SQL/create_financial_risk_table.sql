CREATE TABLE DB_DATALAB_DAF.ML_FINANCIAL_RISK AS (

WITH r1 AS (
SELECT DISTINCT T1.ORDER_PACKAGE_NUMBER, (CASE WHEN T2.IDNT_COMP_FACT IS NULL THEN T3.IDNT_COMP_FACT ELSE T2.IDNT_COMP_FACT END) AS IDNT_COMP_FACTURACTION
FROM VH_S_BANCAIRE.TDI_ECOM_CMD T1

LEFT JOIN vh_b_parc.TDI_KPI_ACCS_FIX T2
ON (T1.ORDER_PACKAGE_NUMBER = T2.ORDER_PACKAGE_NUMBER_COUR)

LEFT JOIN vh_b_parc.TDI_KPI_ACCS_FIX T3
ON (T1.ORDER_PACKAGE_NUMBER = T3.ORDER_PACKAGE_NUMBER)

WHERE IDNT_COMP_FACTURACTION IS NOT NULL
AND Cast(RECORDED_DATE AS DATE) >= DATE_MIN
)

SELECT T1.ORDER_PACKAGE_NUMBER, R1.IDNT_COMP_FACTURACTION AS IDNT_COMP_FACT 
FROM TABLE_DATA T1
LEFT JOIN R1
ON T1.order_package_number = R1.order_package_number
WHERE Cast(T1.RECORDED_DATE AS DATE) BETWEEN DATE_MIN AND DATE_MAX
QUALIFY Count(*) Over (PARTITION BY T1.order_package_number) = 1
) WITH DATA UNIQUE PRIMARY INDEX (ORDER_PACKAGE_NUMBER);