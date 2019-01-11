import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt

# 一、加载数据
data = np.loadtxt('ex1data1.txt', delimiter=',', dtype='float64')
X = data[:, :-1]  # data[:,0:1]
y = data[:, -1:]  # data[:,1:2]
print(X.shape)

plt.scatter(X, y)
# plt.show()

scaler = StandardScaler()
scaler.fit(X)
# print(scaler.mean_) 均值
# print(scaler.scale_) 标准偏差
X = scaler.transform(X)
# print(X)

r_scaler=RobustScaler()
r_scaler.fit(X)
X=r_scaler.transform(X)
# print(X[:,0])
# print(X[:,1])

def plot_after_feature_normalization(X):
    plt.scatter(X,y,color='r')
    plt.show()

plot_after_feature_normalization(X)

model = LinearRegression()
model.fit(X, y)

X_test = np.array([[1650]], dtype='float64')
result = model.predict(X_test)
print(model.coef_)  # Coefficient of the features 决策函数中的特征系数
print(model.intercept_)  # 又名bias偏置,若设置为False，则为0
print(result[0][0])
