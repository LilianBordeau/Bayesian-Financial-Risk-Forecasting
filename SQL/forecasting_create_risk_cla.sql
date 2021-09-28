CREATE SET TABLE DB_DATALAB_DAF.ML_SCORING_FINANCIAL_RISK_CLA (
      ORDER_PACKAGE_NUMBER VARCHAR(255) CHARACTER SET Latin NOT CaseSpecific,
      IDNT_COMP_FACT VARCHAR(50) CHARACTER SET Latin NOT CaseSpecific,
      mtt_rk_01_c FLOAT,
      mtt_rk_02_c FLOAT,
      mtt_rk_03_c FLOAT,
      mtt_rk_04_c FLOAT,
      mtt_rk_05_c FLOAT,
      mtt_rk_06_c FLOAT,
      mtt_rk_07_c FLOAT,
      mtt_rk_08_c FLOAT,
      mtt_rk_09_c FLOAT,
      mtt_rk_10_c FLOAT,
      mtt_rk_11_c FLOAT,
      mtt_rk_12_c FLOAT)
UNIQUE PRIMARY INDEX(ORDER_PACKAGE_NUMBER, IDNT_COMP_FACT);