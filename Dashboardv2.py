import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from enum import Enum
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
dfBIG=pd.read_csv("C:\\Users\\family\\Desktop\\Big12and10.csv")
dfSEC=pd.read_csv("C:\\Users\\family\\Desktop\\SEC.csv")#- For SEC data
dfPAC=pd.read_csv("C:\\Users\\family\\Desktop\\AtlanticCoast.csv")#- For Atlantic Coast and Pac12

df_Predict=pd.read_csv("C:\\Users\\family\\Desktop\\PredictV2.csv")
#plt.scatter(dfBIG['DP'],dfBIG['YDS/GAME'])
 
SecX=dfSEC[['DP','CATCH_%','YAC','YAC/COMP','40YD','REC','TD','YDS/TA','Broad Jump']]# Works for SEC 
BigX=dfBIG[['DP','CATCH_%','YAC','YAC/COMP','40YD','REC','TD','YDS/TA','Broad Jump']] #Works for AtlanticCoast/Pac12 and Big 10/12
PacX=dfPAC[['DP','CATCH_%','YAC','YAC/COMP','40YD','REC','TD','YDS/TA','Broad Jump']] #Works for AtlanticCoast/Pac12 and Big 10/12
#PredictSecX=df_Predict[['DP','CATCH_%','YAC','YAC/COMP','40YD','REC','TARGETS','TD','YDS/TA','Broad Jump']]
PacY=dfPAC['AVG_YDS/SEASON']
SecY=dfSEC['AVG_YDS/SEASON']
BigY=dfBIG['AVG_YDS/SEASON']
PacZ=dfPAC['YDS/GAME']
BigZ=dfBIG['YDS/GAME']
SecZ=dfSEC['YDS/GAME']
PacJ=dfPAC['MAX_YDS/SEASON']
SecJ=dfSEC['MAX_YDS/SEASON']
BigJ=dfBIG['MAX_YDS/SEASON']
PacK=dfPAC['ROOKIE_YDS/GAME']
SecK=dfSEC['ROOKIE_YDS/GAME']
BigK=dfBIG['ROOKIE_YDS/GAME']
# PacK=dfPAC['ROOKIE_YDS']
# SecK=dfSEC['ROOKIE_YDS']
# BigK=dfBIG['ROOKIE_YDS']

# pca=PCA(n_components='mle')
# sc=StandardScaler()
# #SecX= sc.fit_transform(SecX)
# pca.fit(SecX)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)
# #pca.fit_transform()
# principal_C_Sec=pca.fit_transform(SecX)
# #PredictSecX=pca.transform(PredictSecX)
# pcaSecX=pd.DataFrame(data=principal_C_Sec)
# #print(pcaSecX)
regrPac = linear_model.LinearRegression()
regrSec=linear_model.LinearRegression()
regrBig=linear_model.LinearRegression()
regPAC=regrPac.fit(PacX, PacK)
regSEC=regrSec.fit(SecX, SecK)
#SecX=sm.add_constant(SecX)
#regSEC=sm.OLS(SecK,SecX)
#regSEC=regrSec.fit(pcaSecX,SecK)
regBIG=regrBig.fit(BigX,BigK)
#results=regSEC.fit()
#PredictSecX=np.array[df_Predict['DP'],df_Predict['CATCH_%'],]
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

#print (results.rsquared)
# print('AVG_YDS/GAME\n')
print('Intercept: \n', regrSec.intercept_)
print('Coefficients: \n', regrSec.coef_)
#print("R^2: \n",regSEC.score(pcaSecX,SecK))
print("R^2: \n",regSEC.score(SecX,SecK))
# regPAC=regrPac.fit(PacX, PacZ)
# regBIG=regrBig.fit(BigX,BigZ)
# regSEC=regrSec.fit(SecX,SecY)
# print('YDS/GAME\n')
# print('Intercept: \n', regrPac.intercept_)
# print('Coefficients: \n', regrPac.coef_)
# print("R^2: \n",regPAC.score(PacX,PacZ) )
# regPAC=regrPac.fit(PacX,PacJ)
# print('MAX YARDS SEASON \n')
# print('Intercept: \n', regrPac.intercept_)
# print('Coefficients: \n', regrPac.coef_)
# print("R^2: \n",regPAC.score(PacX,PacJ))
