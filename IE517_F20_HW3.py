
import numpy as np
import pandas as pd
import scipy.stats as stats
from pandas import DataFrame
import sys
import pylab
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#import dataset 
df=pd.read_csv('F:\MSFE\IE517 MLF\HY_Universe_corporate bond.csv')
df.head()
print(df.shape)

#Q-Qplot
stats.probplot(df['volume_trades'],dist='norm',plot=pylab)
pylab.show()

#heatmap
#x=df.iloc[:,[22,31,32,33,34]]
#scaler=StandardScaler()
#x_scale=scaler.fit(x)
#x_scale=scaler.transform(x)

#sns.heatmap(x_scale,
            #xticklabels=['volume_trades','weekly_mean_volume','weekly_median_volume','weekly_max_volume','weekly_min_volume'],
            #yticklabels=False )

#plot.show()


#heat map
pd.columns = ['volume_trades','weekly_mean_volume',
              'weekly_median_volume','weekly_max_volume',
              'weekly_min_volume']
corMat = DataFrame(df.iloc[:,[22,31,32,33,34]].corr())
print(corMat)
plot.pcolor(corMat)
plot.show()




#for i in df.columns:
    #if 'Nan' in df[i].tolist():
        #print(i)
        
#moodys=df['Moodys'][-df['Moodys'].isin(['Nan'])]
#print(moodys)

#bloomberg=df['Bloomberg Composite Rating'][-df['Bloomberg Composite Rating'].isin(['Nan'])]
#print(bloomberg)


#scatter plot
plot.grid(True, linestyle='-.')
plot.xlabel('n_trades')
plot.ylabel('volume_trades')
plot.title('n_trades VS volume_trades')
_=plot.scatter(df['n_trades'],df['volume_trades'])


#Histogram
plot.figure(figsize=(15,5))
plot.xlabel('Credit Ratings')
plot.ylabel('Amount')
plot.title('Fitch Histogram')
plot.xticks(fontsize = 10, rotation = 40)
fitch = df['Fitch'][-df['Fitch'].isin(['Nan'])]
_=plot.hist(fitch)

#boxplot
summary = df.describe()
print(summary)
plot.figure(figsize=(5,5))
#array=df.iloc[:,[31,32]].values
abaloneNormal = df.iloc[:,[31,32]]
for i in range(16,17):
    mean = summary.iloc[1,i]
    std = summary.iloc[2,i]
abaloneNormal.iloc[:,i:(i+1)] = (abaloneNormal.iloc[:,i:(i+1)]-mean)/ std  
array= abaloneNormal.values
plot.boxplot(array)


plot.xlabel("Attribute Index")
plot.ylabel("Quartile Ranges - Normalized")
#_=sns.boxplot(array)
plot.show()

print("My name is Ziheng Wu")
print("My NetID is: zihengw5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")






