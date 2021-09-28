CREATE SET TABLE DB_DATALAB_DAF.ML_SCORING_FINANCIAL_RISK_REG (
      ORDER_PACKAGE_NUMBER VARCHAR(255) CHARACTER SET Latin NOT CaseSpecific,
      IDNT_COMP_FACT VARCHAR(50) CHARACTER SET Latin NOT CaseSpecific,
      mtt_rk_01_r FLOAT,
      mtt_rk_02_r FLOAT,
      mtt_rk_03_r FLOAT,
      mtt_rk_04_r FLOAT,
      mtt_rk_05_r FLOAT,
      mtt_rk_06_r FLOAT,
      mtt_rk_07_r FLOAT,
      mtt_rk_08_r FLOAT,
      mtt_rk_09_r FLOAT,
      mtt_rk_10_r FLOAT,
      mtt_rk_11_r FLOAT,
      mtt_rk_12_r FLOAT)
UNIQUE PRIMARY INDEX(ORDER_PACKAGE_NUMBER, IDNT_COMP_FACT);