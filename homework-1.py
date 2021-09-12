import numpy as np
import pandas as pd

# read data
df = pd.read_csv(r'E:/zoomcamp/data.csv')

# no 1 and no 2
# print(np.__version__) #print numpy version
# print(pd.__version__) #print pandas version

# mean (no 3)
# mean = df.groupby('Make').mean() #get mean group by 'Make' Column
# print(mean)  #print mean

# missing values (no 4)
# data = df.loc[df['Year']>=2015] #get data from 2015 or newer
# data.sort_values("Year", inplace = True) #sort data by 'Year'
# print(data.isnull().sum()) #print sum of missing values from sorted data

# fill missing values (no 5)
# mean_hp_before = df['Engine HP'].mean() #get mean from engine HP column
# fill = df.fillna(0) #fill missing values with 0
# mean_hp_after = fill['Engine HP'].mean() #get mean from engine HP column after fill
# print('fill mean= ', round(mean_hp_after)) #print mean after fill
# print('ori mean= ', round(mean_hp_before)) #print mean before fill

# no 6
data_1 = df.loc[df['Make']=='Rolls-Royce'] #get data where 'Make' == 'Rolls-Royce'
data_2 = data_1[['Engine HP', 'Engine Cylinders', 'highway MPG']] #get data only for Engine HP, Engine Cylinders, Highway MPG column
data_3 = data_2.drop_duplicates() #drop duplicated data
x = np.array(data_3) #make array
xt = x.transpose() #transpose array x
XTX = np.dot(xt, x) #multiply array xt and array x
XTXinv = np.linalg.inv(XTX) #invers XTX
# print(np.sum(XTXinv)) #print sum of XTXinv's array element

# no 7
y = np.array([1000, 1100, 900, 1200, 1000, 850, 1300]) #create array y
res = np.dot(XTXinv, xt) #multiply XTXinv and array xt
w = np.dot(res, y) #multiply array res and array y
print(w) #print result (w)