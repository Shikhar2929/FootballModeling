import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from enum import Enum
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.formula.api as smf

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()

    print(selected)
    return model
# def backwardElimination(x, y, sl):
#     numVars = len(x[0])
#     for i in range(0, numVars):
#         regressor_OLS = sm.OLS(y, x).fit()
#         maxVar = max(regressor_OLS.pvalues).astype(float)
#         if maxVar > sl:
#             for j in range(0, numVars - i):
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                     x = (x, j, 1)
#     regressor_OLS.summary()
#     return x
                    




dfBIG=pd.read_csv("C:\\Users\\family\\Desktop\\Big12and10.csv")
dfSEC=pd.read_csv("C:\\Users\\family\\Desktop\\SEC.csv")#- For SEC data
dfPAC=pd.read_csv("C:\\Users\\family\\Desktop\\AtlanticCoast.csv")#- For Atlantic Coast and Pac12

df_Predict=pd.read_csv("C:\\Users\\family\\Desktop\\PredictV2.csv")
#plt.scatter(dfBIG['DP'],dfBIG['YDS/GAME'])
 
SecX=dfSEC[['DP','CATCH_P','YAC','YAC_COMP','FORTYYD','REC','TD','YDS_TA','BroadJump','TARGETS','ROOKIE_YDS_GAME']]# Works for SEC 
BigX=dfBIG[['DP','CATCH_P','YAC','YAC_COMP','FORTYYD','REC','TD','YDS_TA','BroadJump','TARGETS','ROOKIE_YDS_GAME']] #Works for AtlanticCoast/Pac12 and Big 10/12
#PacX=dfPAC[['DP','CATCH_P','YAC','YAC_COMP','FORTYYD','REC','TD','YDS_TA','BroadJump','TARGETS','ROOKIE_YDS_GAME']] #Works for AtlanticCoast/Pac12 and Big 10/12
PacX=dfPAC[['DP','CATCH_P','YAC','YAC_COMP','FORTYYD','REC','TD','YDS_TA','BroadJump','TARGETS']] #Works for AtlanticCoast/Pac12 and Big 10/12

#PacX=dfPAC[['DP','CATCH_%','40YD','REC','TD','YDS/TA','TARGETS']] #Works for AtlanticCoast/Pac12 and Big 10/12

#PredictSecX=df_Predict[['DP','CATCH_%','YAC','YAC/COMP','40YD','REC','TARGETS','TD','YDS/TA','Broad Jump']]
PacY=dfPAC['AVG_YDS_SEASON']
SecY=dfSEC['AVG_YDS_SEASON']
BigY=dfBIG['AVG_YDS_SEASON']
PacZ=dfPAC['YDS_GAME']
BigZ=dfBIG['YDS_GAME']
SecZ=dfSEC['YDS_GAME']
PacJ=dfPAC['MAX_YDS_SEASON']
SecJ=dfSEC['MAX_YDS_SEASON']
BigJ=dfBIG['MAX_YDS_SEASON']
PacK=dfPAC['ROOKIE_YDS_GAME']
SecK=dfSEC['ROOKIE_YDS_GAME']
BigK=dfBIG['ROOKIE_YDS_GAME']
# PacK=dfPAC['ROOKIE_YDS']
# SecK=dfSEC['ROOKIE_YDS']
# BigK=dfBIG['ROOKIE_YDS']
# model=forward_selected(SecX,'ROOKIE_YDS')
# print(model)
# regrPac = linear_model.LinearRegression()
# regrSec=linear_model.LinearRegression()
# regrBig=linear_model.LinearRegression()
# regPAC=regrPac.fit(PacX, PacK)
# regSEC=regrSec.fit(SecX, SecK)
# SecX=sm.add_constant(SecX)
# regSEC=sm.OLS(SecK,SecX)
# regBIG=sm.OLS(BigK,BigX)
regPAC=sm.OLS(PacK,PacX)
# resultsSEC=regSEC.fit()
resultsPAC=regPAC.fit()
SecX=SecX.to_numpy()
SecY=SecY.to_numpy()
model=backwardElimination(SecX,SecY,0.05)
print(model)
# resultsBIG=regBIG.fit()
#model=forward_selected(PacX,'ROOKIE_YDS_GAME')

# for i in df_Predict.index:
#     print(df_Predict['Conference'][i])
#     if df_Predict['Conference'][i]=='Southeastern':
#         print(df_Predict['Player'][i])
#         pred=regrSec.predict([[df_Predict['DP'][i],df_Predict['CATCH_P'][i],df_Predict['YAC'][i],df_Predict['YAC_COMP'][i],df_Predict['40YD'][i],df_Predict['REC'][i],df_Predict['TD'][i],df_Predict['YDS/TA'][i],df_Predict['Broad Jump'][i]]])
#         if pred<0:
#             pred=0
#         print('Predicted AVG_YDS/SEASON: \n', pred)
#     if df_Predict['Conference'][i]=='Big':
#         print(df_Predict['Player'][i])
#         print('Predicted AVG_YDS/SEASON: \n', regrBig.predict([[df_Predict['DP'][i],df_Predict['CATCH_P'][i],df_Predict['YAC'][i],df_Predict['YAC_COMP'][i],df_Predict['40YD'][i],df_Predict['REC'][i],df_Predict['TD'][i],df_Predict['YDS/TA'][i],df_Predict['Broad Jump'][i]]]))
#     if df_Predict['Conference'][i]=='Pac-12':
#         print(df_Predict['Player'][i])
#         pred=regrPac.predict([[df_Predict['DP'][i],df_Predict['CATCH_P'][i],df_Predict['YAC'][i],df_Predict['YAC_COMP'][i],df_Predict['40YD'][i],df_Predict['REC'][i],df_Predict['TD'][i],df_Predict['YDS/TA'][i],df_Predict['Broad Jump'][i]]])
#         if pred<0:
#             pred=0
#         print('Predicted AVG_YDS/SEASON: \n', pred)

# print (resultsSEC.rsquared_adj)
# print(resultsSEC.summary())
#print (resultsPAC.rsquared_adj)
# print (resultsBIG.rsquared_adj)
# print(model.summary())
#print(model.rsquared_adj)
# print('AVG_YDS/GAME\n')
#print('Intercept: \n', regrSec.intercept_)
#print('Coefficients: \n', regrSec.coef_)
#print("R^2: \n",regSEC.score(pcaSecX,SecK))
#print("R^2: \n",regSEC.score(SecX,SecK))
# regPAC=regrPac.fit(PacX, PacZ)
# regBIG=regrBig.fit(BigX,BigZ)
# regSEC=regrSec.fit(SecX,SecY)
# print('YDS/GAME\n')
# print('Intercept: \n', regrPac.intercept_)
# print('Coefficients: \n', regrPac.coef_)
# print("R^2: \n",regPAC.score(PacX,PacZ) )
# regPAC=regrPac.fit(PacX,PacJ)
