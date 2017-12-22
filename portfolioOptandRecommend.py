

import requests

url = "http://107.20.102.244:4000/api/Analysis"

payload = "{  \n   \"connectionMode\":\"Database\",\n   \"datasource\":\"user\",\n   \"tags\":null,\n   \"constraintParameters\":null,\n   \"analysePortfolios\":true,\n   \"simulatedReturns\":null,\n   \"portfolios\":[  \n      {  \n         \"portfolioId\":\"d5dd79579617e848913a151469c74\",\n         \"portfolioName\":\"John Doe's Portfolio\",\n         \"positionType\":\"shares\",\n         \"PortfolioType\":\"SharesAndCashAmount\",\n         \"totalPortfolioValue\":200,\n         \"amountType\":\"cash\",\n         \"amountValue\":3000,\n         \"rfr\":0.006,\n         \"currency\":\"USD\",\n         \"targetLevel\":\"Conservative\",\n         \"positions\":[  \n            {  \n               \"positionId\":1162,\n               \"instrument\":{  \n                  \"instrumentId\":\"US9229087690\"\n               },\n               \"position\":2000,\n               \"sector\":\"Financial\",\n               \"transaction_cost\":0\n            },\n            {  \n               \"positionId\":1163,\n               \"instrument\":{  \n                  \"instrumentId\":\"US9220428588\"\n               },\n               \"position\":2500,\n               \"sector\":\"Financial\",\n               \"transaction_cost\":0\n            },\n            {  \n               \"positionId\":1164,\n               \"instrument\":{  \n                  \"instrumentId\":\"US9219438580\"\n               },\n               \"position\":500,\n               \"sector\":\"Financial\",\n               \"transaction_cost\":0\n            }\n         ]\n      }\n   ],\n   \"returnmethod\": \"HistoricalTrend\",\n   \"hedge\":false,\n   \"filteringlevel\":0.1,\n   \"estimationwindow\":252,\n   \"returnHorizon\":252,\n   \"riskhorizon\":252,\n   \"riskmeasure\":\"Volatility\"\n}\n"
headers = {
    'content-type': "application/json",
    'cache-control': "no-cache",
    'postman-token': "ec283624-a9ef-e50f-0160-7cea91ea3dcd"
    }

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)


# We want to adress this analysis results, so we convert it into JSON file

# In[3]:

import pandas as pd
from pandas_datareader import data
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


# In[5]:

response.json()[0] #type is dictionary


# <p>We record these response results in the JSON:</p>
# <p>Weights, risk, return, sharpeRatio, deiversification, exanteMaxDrawDown</p>

# In[6]:

from pandas import DataFrame, Series
import pandas as pd


# In[7]:

response.json()[0]['portfolioResults']


# In[8]:

frame = DataFrame(response.json())
frame


# In[9]:

data={'Weights':response.json()[0]['weights'],
      'Risk':response.json()[0]['portfolioResults']['risk'],
      'Return':response.json()[0]['portfolioResults']['return'],
      'Sharpe Ratio':response.json()[0]['portfolioResults']['sharpeRatio'],
      'Deiversification':response.json()[0]['portfolioResults']['diversification'],
      'Exante MaxDrawDown':response.json()[0]['portfolioResults']['exanteMaxDrawDown']}


# In[10]:

frame_data=DataFrame(data)


# In[11]:

frame_data


# In[12]:

response.json()[0]['weights']


# In[13]:

data={'Weights':frame['weights'],
      'Risk':frame['portfolioResults'][0]['return']}
data={'Weights':frame['weights'],
      'Risk':frame['portfolioResults'][0]['risk'],
      'Return':frame['portfolioResults'][0]['return'],
      'Sharpe Ratio':frame['portfolioResults'][0]['sharpeRatio'],
      'Deiversification':frame['portfolioResults'][0]['diversification'],
      'Exante MaxDrawDown':frame['portfolioResults'][0]['exanteMaxDrawDown']}
#frame_data1=DataFrame(frame['weights'],frame['portfolioResults'][0]['return'])
frame_data1 = DataFrame(data)
frame_data1


# In[14]:

frame['portfolioResults'][0]


# <h4>Now, we perform portfolio optimization:</h4>


# In[15]:

import requests

url = "http://107.20.102.244:4000/api/Optimize"

payload = "{  \n   \"connectionMode\":\"Database\",\n   \"datasource\":\"user\",\n   \"tags\":null,\n   \"constraintParameters\":null,\n   \"simulatedReturns\":null,\n   \"analysePortfolios\":true,\n   \"portfolios\":[  \n      {  \n         \"portfolioId\":\"742a95cdeb2e27305a5be46664eab\",\n         \"portfolioName\":\"John Doe`s portfolio\",\n         \"positionType\":\"shares\",\n         \"PortfolioType\":\"SharesAndCashAmount\",\n         \"totalPortfolioValue\":200,\n         \"amountType\":\"cash\",\n         \"amountValue\":3000,\n         \"rfr\":0.006,\n         \"currency\":\"USD\",\n         \"targetLevel\":\"Conservative\",\n         \"positions\":[  \n            {  \n               \"positionId\":705,\n               \"instrument\":{  \n                  \"instrumentId\":\"US9229087690\"\n               },\n               \"position\":2000,\n               \"sector\":\"Financial\",\n               \"transaction_cost\":0\n            },\n            {  \n               \"positionId\":706,\n               \"instrument\":{  \n                  \"instrumentId\":\"US9219438580\"\n               },\n               \"position\":500,\n               \"sector\":\"Financial\",\n               \"transaction_cost\":0\n            },\n            {  \n               \"positionId\":707,\n               \"instrument\":{  \n                  \"instrumentId\":\"US9220428588\"\n               },\n               \"position\":2500,\n               \"sector\":\"Financial\",\n               \"transaction_cost\":0\n            },\n            {  \n               \"instrument\":{  \n                  \"instrumentId\":\"US22542D8781\"\n               },\n               \"positionId\":9999,\n               \"sector\":\"Financial\",\n               \"position\":1,\n               \"transaction_cost\":0\n            }\n         ]\n      }\n   ],\n   \"strategy\":{  \n      \"strategyName\":\"Strategy Name\",\n      \"description\":null,\n      \"filteringLevel\":0.1,\n      \"hedge\":false,\n      \"riskHorizon\":252,\n      \"returnHorizon\":252,\n      \"estimationWindow\":120,\n      \"returnMethod\":\"HistoricalTrend\",\n      \"optimisationModel\":\"minrisk\",\n      \"riskMeasure\":\"Volatility\",\n      \"targetCstType\":\"Performance\",\n      \"turnoverCst\":0.7,\n      \"portfolioTransactionCostCst\":null,\n      \"annualPerfTargetCst\":0.155,\n      \"annualBudget\":1,\n      \"instrumentsBoxConstraints\":{  \n         \"cash\":{  \n            \"min\":0,\n            \"max\":1\n         }\n      },\n      \"sectorBoxConstraints\":{  \n\n      }\n   }\n}\n\n\n"
headers = {
    'content-type': "application/json",
    'cache-control': "no-cache",
    'postman-token': "90675e11-b474-9f70-67cf-bcb2ee0b8210"
    }

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)


# In[16]:

payload = payload.replace('\n','').replace(' ','')


# In[17]:

payload_turnoverCst=payload.replace('"turnoverCst":0.7','"turnoverCst":toc_input')


# <h4>Now, we test different TurnoverCst Value in the Portfolio Optimization Process:</h4>

# In[18]:

import requests

url = "http://107.20.102.244:4000/api/Optimize"

headers = {
    'content-type': "application/json",
    'cache-control': "no-cache",
    'postman-token': "90675e11-b474-9f70-67cf-bcb2ee0b8210"
    }

toc = np.arange(0,2.1,0.1)

payload_rep = payload_turnoverCst.replace('"turnoverCst":0.7','"turnoverCst":toc_input')

response_list = []

for i in range(0,len(toc)):

    payload = payload_rep.replace('toc_input',str(toc[i]))
    response = requests.request("POST", url, data=payload, headers=headers)
    response_list.append(response.json())


# Put 21 times response results in to list and made into DataFrame

# In[31]:

response_list[0][0]['analysisResponse']['portfolioResults']


# In[20]:

toc_list = []
for i in range(0,len(toc)):
    toc_data = {}
    toc_data.update({'TurnoverCst' : toc[i]})
    toc_data.update({'Cash Weight' : response_list[i][0]['analysisResponse']['weights']['cash']})
    toc_data.update({'CSLS Weight' : response_list[i][0]['analysisResponse']['weights']['uS22542D8781']})
    toc_data.update({'VEA Weight' : response_list[i][0]['analysisResponse']['weights']['uS9219438580']})
    toc_data.update({'VWO Weight' : response_list[i][0]['analysisResponse']['weights']['uS9220428588']})
    toc_data.update({'VTI Weight' : response_list[i][0]['analysisResponse']['weights']['uS9229087690']})
    
    toc_list.append(toc_data)

frame_toc = DataFrame(toc_list)


# In[32]:

toc_ana_list = []
for i in range(0,len(toc)):
    toc_ana_data = {}
    toc_ana_data.update({'TurnoverCst' : toc[i]})
    toc_ana_data.update({'Return' : response_list[i][0]['analysisResponse']['portfolioResults']['return']})
    toc_ana_data.update({'Risk' : response_list[i][0]['analysisResponse']['portfolioResults']['risk']})
    toc_ana_data.update({'SharpeRatio' : response_list[i][0]['analysisResponse']['portfolioResults']['sharpeRatio']})
    
    toc_ana_list.append(toc_ana_data)

frame_toc_ana = DataFrame(toc_ana_list)




# In[37]:

frame_toc_ana.set_index('TurnoverCst')


# In[27]:

frame_toc.set_index('TurnoverCst').plot(kind='bar',figsize=(12,8),stacked=True,ylim=(0,1.25))


# In[38]:

import requests

url = "http://107.20.102.244:4000/api/Analysis"

payload = "{  \n   \"connectionMode\":\"Database\",\n   \"datasource\":\"user\",\n   \"tags\":null,\n   \"constraintParameters\":null,\n   \"analysePortfolios\":true,\n   \"simulatedReturns\":null,\n   \"portfolios\":[  \n      {  \n         \"portfolioId\":\"d5dd79579617e848913a151469c74\",\n         \"portfolioName\":\"John Doe's Portfolio\",\n         \"positionType\":\"shares\",\n         \"PortfolioType\":\"SharesAndCashAmount\",\n         \"totalPortfolioValue\":200,\n         \"amountType\":\"cash\",\n         \"amountValue\":3000,\n         \"rfr\":0.006,\n         \"currency\":\"USD\",\n         \"targetLevel\":\"Conservative\",\n         \"positions\":[  \n            {  \n               \"positionId\":1162,\n               \"instrument\":{  \n                  \"instrumentId\":\"US9229087690\"\n               },\n               \"position\":2000,\n               \"sector\":\"Financial\",\n               \"transaction_cost\":0\n            },\n            {  \n               \"positionId\":1163,\n               \"instrument\":{  \n                  \"instrumentId\":\"US9220428588\"\n               },\n               \"position\":2500,\n               \"sector\":\"Financial\",\n               \"transaction_cost\":0\n            },\n            {  \n               \"positionId\":1164,\n               \"instrument\":{  \n                  \"instrumentId\":\"US9219438580\"\n               },\n               \"position\":500,\n               \"sector\":\"Financial\",\n               \"transaction_cost\":0\n            }\n         ]\n      }\n   ],\n   \"returnmethod\": \"HistoricalTrend\",\n   \"hedge\":false,\n   \"filteringlevel\":0.1,\n   \"estimationwindow\":252,\n   \"returnHorizon\":252,\n   \"riskhorizon\":252,\n   \"riskmeasure\":\"Volatility\"\n}\n"
headers = {
    'content-type': "application/json",
    'cache-control': "no-cache",
    'postman-token': "ec283624-a9ef-e50f-0160-7cea91ea3dcd"
    }

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)


# In[39]:

response.json()[0] #type is dictionary


# In[43]:

def eff_plot(etfs,weightslist):
    weights = np.array([float(x.strip()) for x in weightslist])
    ticker_data = etf_data[etfs]
    returns = ticker_data.pct_change()
    
    # weight type transform:
    temp = [None]*len(etfs)
    for i in range(len(etfs)):
        temp[i] = ticker_data.iloc[-1][etfs[i]]
    latest_price = np.array(temp)
    weights0 = weights*latest_price
    weights0 /= np.sum(weights0)
    
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # calculate original portfolio:
    portfolio_return = np.sum(mean_daily_returns * weights0) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights0.T,np.dot(cov_matrix, weights0))) * np.sqrt(252)
    portfolio_sharpe = portfolio_return / portfolio_std_dev
    
    test0=[None]*len(etfs)
    for i in range(len(etfs)):
        test0[i] = weights0[i]
    original=[portfolio_return, portfolio_std_dev, portfolio_sharpe]
    original.extend(test0)

    test1=[None]*len(etfs)
    for i in range(len(etfs)):
        test1[i]=etfs[i]
    index=['ret','stdev','sharpe']
    index.extend(test1)

    original_portfolio = pd.DataFrame({'Original_Portfolio': original},index=index)
    
    # Monte Carlo Simulation to run 30,000 runs of different randomly generated weights
    num_trials = 30000
    results = np.zeros((3+len(etfs),num_trials))
    
    for i in range(num_trials):
        weights = np.array(np.random.random(len(etfs)))
        weights /= np.sum(weights)

        test_ret = np.sum(mean_daily_returns * weights) * 252
        test_std = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)

        results[0,i] = test_ret
        results[1,i] = test_std
        results[2,i] = results[0,i] / results[1,i]

        for j in range(len(weights)):
            results[j+3,i] = weights[j]

    temp0=[None]*len(etfs)
    for i in range(len(etfs)):
        temp0[i]=etfs[i]
    columns=['ret','stdev','sharpe']
    columns.extend(temp0)
    results_frame = pd.DataFrame(results.T,columns=columns)  

    #locate position of portfolio with highest Sharpe Ratio
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
    #locate positon of portfolio with minimum standard deviation
    min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
    
    #Plot Setting
    plt.figure(figsize=(15,9))
    plt.title("Portfolio Optimization", size=25)

    plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu',label='_nolegend_')
    plt.xlabel('Risk Index',size=16)
    plt.ylabel('Returns',size=16)
    plt.colorbar()

    #plot red star to highlight position of portfolio with highest Sharpe Ratio
    plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=600,label='Max Sharpe')
    #plot blue star to highlight position of minimum variance portfolio
    plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='b',s=600,label ='Min Risk')
    #plot green star to highlight position of original portfolio
    plt.scatter(portfolio_std_dev,portfolio_return,marker=(5,1,0),color='green',s=600,label='Original')
    plt.legend(loc='best',prop={'size': 15})
    
    RS_frame = pd.DataFrame({'Min_Risk':min_vol_port, 'Max_Sharpe_Ratio':max_sharpe_port})
    MCS_result = pd.concat([RS_frame,original_portfolio],axis=1)

    return pd.concat([RS_frame,original_portfolio],axis=1)
