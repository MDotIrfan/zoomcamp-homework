import numpy as np
import pandas as pd

# read data
df = pd.read_csv(r'E:/zoomcamp/data.csv')

# no 1 and no 2
# print(np.__version__)
# print(pd.__version__)

# mean (no 3)
# mean = df.groupby('Make').mean()
# print(mean)

# missing values (no 4)
# data = df.loc[df['Year']>=2015]
# data.sort_values("Year", inplace = True)
# print(data.isnull().sum())

# fill missing values (no 5)
# mean_hp_before = df['Engine HP'].mean()
# fill = df.fillna(0)
# mean_hp_after = fill['Engine HP'].mean()
# print('fill mean= ', round(mean_hp_after))
# print('ori mean= ', round(mean_hp_before))

# no 6
data_1 = df.loc[df['Make']=='Rolls-Royce']
data_2 = data_1[['Engine HP', 'Engine Cylinders', 'highway MPG']]
data_3 = data_2.drop_duplicates()
x = np.array(data_3)
xt = x.transpose()
XTX = np.dot(xt, x)
XTXinv = np.linalg.inv(XTX)
# print(np.sum(XTXinv))

# no 7
y = np.array([1000, 1100, 900, 1200, 1000, 850, 1300])
res = np.dot(XTXinv, xt)
w = np.dot(res, y)
print(w)