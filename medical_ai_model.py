import pandas as pd
from pandas import Categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#读取心血管疾病数据库
data = pd.read_csv('cardio_train.csv', sep = ';')

#第二步：进行数据的初了解以及数据清洗
# print(data.info())
# print(data.head()) 
# print(data.isnull().sum()) 

#由于id对于数据预测没有作用所以去除id列，并且去重
data.drop('id', axis = 1, inplace = True)
data.drop_duplicates(inplace= True)

#将年龄由天改成年
data['age'] = data['age'] // 365

#去除年龄过大和过小的人
out_filter = ((data['age'] < 18) | (data['age'] > 70))
data = data[~out_filter]

#去除测量的收缩舒张压有问题的数据
out_filter2 = ((data['ap_hi'] < 0) | (data['ap_lo'] < 0))
data = data[~out_filter2]

#去除收缩压舒张压过高的数据
out_filter3 = ((data['ap_hi'] > 250) | (data['ap_lo'] > 200))
data = data[~out_filter3]

#去除逻辑错误的,即收缩压大于舒张压
out_filter4 = (data['ap_hi'] < data['ap_lo'])
data = data[~out_filter4]

#第三步：划分特征集(X，除去cardio的特征)和标签集(y，也就是cardio)，并且数据划分为训练集和测试集
X = data.drop('cardio', axis = 1)
y = data['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size= 0.2,
                                                    random_state= 42)
print(f'训练集数据大小：{len(X_train)}')
print(f'测试集数据大小：{len(X_test)}')

#标准化
#数值型特征
numeric_col = ['age', 'height', 'ap_hi', 'ap_lo']

#分类型特征
categorical_col = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

#数值型特征标准化
scaler = StandardScaler()
X_train[numeric_col] = scaler.fit_transform(X_train[numeric_col])
X_test[numeric_col] = scaler.fit_transform(X_test[numeric_col])