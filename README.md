## 一、Introduction
### 1.监督学习(supervised learning)
    监督学习被分类为回归问题和分类问题。
    在回归问题中，我们以连续的输出值为预测结果；
    在分类问题中，我们以离散的输出值为预测结果。
### 2.非监督学习(Unsuperivised learning)
    我们通过数据中变量之间的关系来对数据进行聚类。
### 3.模型表示(Model Representation)
![coursera1](https://note.youdao.com/yws/api/personal/file/WEB51a9035af62b47cbb0352ed7323953fe?method=getImage&version=6122&cstk=AocfnfFN)
### 4.损失函数(Cost Function)
![coursera2](https://note.youdao.com/yws/api/personal/file/WEBc4965f77e818e4dd8fe5c2a2500a8724?method=getImage&version=6129&cstk=AocfnfFN)
### 5.梯度下降(Parameter learning--Gradient Descent)
## 三、逻辑回归(Logistic Regression)
    逻辑回归：将数据分类为离散值
### 3.1 分类和表示(Classification & Representation)
#### 1.分类
    分类的一种方法是使用线性回归并将大于0.5的结果，预测为1；
    将所有小于0.5的结果，预测为0；
    但是，此方法不能很好地工作，因为分类实际上不是线性函数。
#### 2.假设表示(Hypothesis Representation)
![image](https://gypsy-1255824480.cos.ap-beijing.myqcloud.com/ml/logistic1.jpg)

取值范围:$$0 <= h_\theta(x) <=1 $$

![image](https://gypsy-1255824480.cos.ap-beijing.myqcloud.com/ml/logistic2.jpg)
#### 3.决策边界(Decision Boundary)
![image](https://gypsy-1255824480.cos.ap-beijing.myqcloud.com/ml/logistic3.jpg)

$$ h_\theta(x)>=0.5$$ --> y=1  
$$ h_\theta(x)< 0.5$$ --> y=0

$$ g(Z)>=0.5$$ --> when Z>=0  
$$ h_\theta(x)=g(\theta^Tx)>= 0.5$$ --> whenever $$\theta^Tx>=0$$  
$$ g(Z)<0.5$$ --> when Z<0  
$$ h_\theta(x)=g(\theta^Tx)< 0.5$$ --> whenever $$\theta^Tx<0 $$

![image](https://gypsy-1255824480.cos.ap-beijing.myqcloud.com/ml/e.jpg)

![image](https://gypsy-1255824480.cos.ap-beijing.myqcloud.com/ml/e1.jpg)
### 3.2 逻辑回归模型(Logistic Regression Model)
### 1.损失函数(Cost Function)
    在逻辑回归中，我们不能再使用线性回归的损失函数了，因为逻辑函数会造成输出波动，产生很多局最优，即它不是一个凸函数。

![image](https://gypsy-1255824480.cos.ap-beijing.myqcloud.com/ml/logistic4.jpg)

函数推倒：
![image](https://gypsy-1255824480.cos.ap-beijing.myqcloud.com/ml/logistic5.png)

![image](https://gypsy-1255824480.cos.ap-beijing.myqcloud.com/ml/logistic6.jpg)
#### 2.简化损失函数和梯度下降(Simplified Cost Function and Gradient Descent)
$$ Cost(h_\theta(x),y)=-ylog(h_\theta(x))-(1-y)log(1-h_\theta(x))$$

![image](https://gypsy-1255824480.cos.ap-beijing.myqcloud.com/ml/logistic7.jpg)
#### 3.高级优化(Advanced Optimization)
    Conjugate gradient
    BFGS
    L-BFGS 
    
```
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

```
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```
### 3.3 多分类(Multiclass Classification)
![image](https://gypsy-1255824480.cos.ap-beijing.myqcloud.com/ml/logistic8.jpg)

    我们将多分类问题看作为多个二分类问题，并对每个case进行预测，将获得的最大概率值来选择类别。
### 3.4 过拟合




    
    



