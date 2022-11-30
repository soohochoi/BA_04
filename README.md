# BA_04 Ensemble Learning(Bagging & Boosting)

<p align="center"><img width="604" alt="image" src="https://user-images.githubusercontent.com/97882448/204542304-85e7f83d-075c-4d02-b5e8-6e8d77d66987.png">

이자료는 고려대학교 비니지스 애널리틱스 강필성교수님께 배운 Ensemble Learning(Bagging & Boosting)을 바탕으로 만들어졌습니다.
먼저, Ensemble Learning(Bagging & Boosting)방식의 기본적인 개념을 배운후에 Bagging에선 Random Forest,Boosting에선 XG Boost에 대해 개념 및 코드를 통해 직접 구현을 해봄으로써 이해를 돕도록하겠습니다. 또한 Auto ML과의 성능차이를 비교해 보며 간단한 실험을 해보겠습니다.

## 목차
### 1. [BA_04 Ensemble Learning_개념설명](#ba_04-ensemble-learning_개념설명)
### 2. [Bagging_Random Forest(RF) 개념설명](#bagging_random-forestrf-개념설명)
### 3. [Boosting_XGboost(xgb) 개념설명](#boosting_xgboostxgb-개념설명)
### 4. [Ensemble Learning에서 Random_forest와 XG boost 실습코드](#ensemble-learning에서-random_forest와-xg-boost-실습코드)
  #### 4.1. [데이터 셋 소개](#데이터-셋-소개)
  #### 4.2. [코드설명](#코드설명)
  #### 4.2.1 [random-forest](#random-forest)
  #### 4.2.2 [XGboost](#xgboost)
  #### 4.2.3 [pycaret_RF](#pycaret_rf)
  #### 4.2.4 [pycaret_xgboost](#pycaret_xgboost)
  #### 4.3. [결론](#결론)
  
## BA_04 Ensemble Learning_개념설명
Ensemble Learning 이란 여러 개의 분류기(Classifier)를 생성하고 그 예측을 결합함으로써 보다 정확한 예측을 도출하는 기법을 말합니다. 그러면 왜? 이 앙상블 러닝을 사용하는지 궁금하실수 있습니다.  이유는 먼저 하나의 Single 모델이 모든 데이터셋에서 우수하지 않습니다.  그래서 여러 다양한 학습 알고리즘들을 결합하여 학습시키는 방법을 고안하였습니다. 따라서 예측력의 보완되고  각각의 알고리즘을 single로 사용할 경우 나타나는 단점들을 보완되어집니다. 쉽게 말해서 옛 속담에 백지장도 맞들면 낫다와 같이 집단 지성을 이용하는 방법입니다.
그러면 궁금증이 많은 몇분께서는 어떻게 증명할건데? 라고 하실겁니다. 먼저는 아래 그림처럼 논문에서 실제 데이터셋에 대하여 증명이 되었습니다. 
<p align="center"><img width="922" alt="image" src="https://user-images.githubusercontent.com/97882448/204577328-c787f716-a71c-401b-bef3-87a01268a0e4.png">
  
2014년에 publish된 논문을 참조하면 single모델보다 Ensemble모델이 더 나은 성능을 보임을 입증하였습니다.
위 논문은 121개의 데이터셋에서 179개의 알고리즘의 성능을 평가하였습니다. 물론 179개의 알고리즘안에서 파이썬, R과 같은 컴퓨터언어가 달라도 다른 알고리즘이라고 분류하여 실험하였고 곧 배울 GBM, light GBM, Cat Boost와 같은  Boosting 계열 모델은 사용되지 않았지만 결과적으로 Ensemble에 Bagging 계열의 Random Forests (의사 결정 나무)가 32.9등 으로 가장 성능이 좋음을 보였습니다. 이를 통해 모든 데이터 셋에서 뛰어난 1등 알고리즘은 존재하지 않지만 Ensemble모델이 single모델보단 거의 대부분 좋음을 보임을 입증하였습니다.

<p align="center"><img width="530" alt="image" src="https://user-images.githubusercontent.com/97882448/204579735-7f79453a-749c-4d3f-befd-95639f9c65df.png">
  
또한 실제 데이터로 모델을 만들때에는 빨간 박스처럼 항상 노이즈(𝜖)가 있습니다. 그래서 이 노이즈를 최대한 줄여 모델에 의한 오류를 최대한 줄이는게 좋은 모델입니다. 모델에 의한 오류를 정리하면 Bias(편향)와 variance(분산) 그리고 우리가 제거 할수 없는 시그마의 제곱이 있습니다. 
* Bias(편향): 반복적인 모델 학습 시 평균적으로 얼마나 정확한 추정이 가능한지에 대한 측정 지표
* variance(분산): 반복적인 모델 학습 시 개별추정이 얼마나 차이가 크게 나타나는지에 대한 측정 지표
조금더 편향과 분산이 무엇인지 알아보겠습니다. 
<p align="center"><img width="485" alt="image" src="https://user-images.githubusercontent.com/97882448/204589240-dad55a14-d4f4-480c-897f-5ea4c78221a4.png">

위에 그림에서 4사분면부터 보면 편향과 분산이 둘 다 높은 것을 볼 수 있음 이것은 이것은 도태된 모델이라고 할 수 있습니다.
3사분면은 편향은 높지만 분산은 낮은 경우를 말합니다. 예를 들어 logistic regression이나  K-nn에서 K가 클 때를 말합니다. 2사분면은 편향은 낮지만 분산은 높은 경우립니다. 예를 들어 Decision Tree나  K-nn에서 K가 작을 때를 말합니다. 이토록 현실에선 3사분면과 2사분면 같은 모델이 대부분입니다. 따라서 1사분면같이 편향과 분산이 둘 다 낮아야 좋은 모델을 충족시키기 위해 Ensemble방식이 제안되어집니다. 2사분면의 분산을 줄여줘서 1사분면처럼 만드는 방식이 Bagging이고 3사분면의 편향을 줄여줘서 1사분면처럼 만드는 방식이 Boosting입니다. 
마지막으로, 앙상블을 구성할때 2가지 핵심 아이디어까지 알아보겠습니다. 앙상블을 구성할때 90%정도 가장 중요한것은 다양성을 확보해야한다는 것입니다. 즉 동일한 모델을 여러 개 사용하는 것은 아무런 효과가 없으므로 개별적으로 어느정도 다르고 좋은 성능을 가지면서 앙상블 내에서 각 모델이 서로 다양한 형태로 나타내줘어야 합니다. 나머지 10%는 결과물을 잘 결합해야한다는것입니다. 

## Bagging_Random Forest(RF) 개념설명
<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/97882448/204602601-e8a93bc9-dcc6-4438-b744-81bf63fa97d7.png">

RF는 의사 결정 나무를 사용하는 배깅(Bagging)의 특수한 형태입니다.다수의 나무가 모여서 forest을 이루어 랜덤포레스트라고 불립니다.
원래 의사결정 나무는 결정 경계를 탐색할 때 모든 변수를 고려하여 분기점을 찾지만 Random forest는 이 중 일부의 변수만을 랜덤하게 선택하여 분기점을 찾습니다. 그러면 이글을 읽는 사람은 반문할수 있을겁니다. Random하게 선택된 변수는 전체변수를 고려 하는 것 보다 의사 결정 나무의 성능이 떨어질 수 밖에 없는데 어떻게 앙상블 방식이 좋은 성능이 나올수 있니? 라고요. 하지만 다양성이 고려되고 매번 다른 특성을 고려하기 때문에 훨씬 더 다양한 모델을 만들 수 있습니다. 이러한 RF의 장점과 단점은 아래와 같습니다. 

- Random Forest의 장점 
    - Classification 및 Regression 문제에 모두 사용 가능
    - 대용량 데이터 처리에 효과적
    - 과대적합 문제 최소화하여 모델의 정확도 향상
    - Classification 모델에서 상대적으로 중요한 변수를 선정 및 Ranking 가능
- Random Forest의 단점 
    - 랜덤 포레스트 특성상 데이터 크기에 비례해서 수백개에서 수천개의 트리를 형성하기에 예측에 오랜 프로세스 시간이 걸림
    - 랜덤 포레스트 모델 특성상 생성하는 모든 트리 모델을 다 확인하기 어렵기에 해석 가능성이 떨어질수도 있음

## Boosting_XGboost(xgb) 개념설명
<p align="center"><img width="700" alt="image" src="https://user-images.githubusercontent.com/97882448/204880494-8e40ed4e-dd62-4dc6-8481-234fc1d418ff.png">

XGBoost는 의사결정나무의 Boosting기반 모델로써 Extreme Gradient Boosting이 원어이다. 효율성과 유연성이 뛰어나며 Gradient Boosting framework의 뿌리에서 나왔다. 또한 여러 하이퍼 파라미터를 조절해가면서 최적의 Model을 만들 수 있고 Over fitting(과적합)을 방지할 수 있다. 자원이 많아도 빠르게 학습및 예측이 가능한것도 장점이며 효율적인 결측치 데이터 처리와 Cross validation을 지원하는 특징들이 있다.
  
구조를 조금더 자세히 이야기하자면 병렬화된 tree 구조 생성한다. XGBoost는 block라고 부르는 곳에 데이터를 저장하는데 각 block의  데이터는  compressed column(CSC) format으로 저장되며, 각 칼럼은 해당 변수 값을 정렬한 상태로 저장된다. 보통 다른 알고리즘은 정렬 정보 보존이 안되는데 이 부분은 XG boost가 시간을 보존하는 방법이다. 또한 깊이 우선탐색을 통한 분기점 생성한다. 좌측에서 부터 우측으로 변수 값이 오름차순으로 정렬되어 있다고 가정하면 전체 데이터를 균등하게 분할하고 각 분할에 대해 개별적으로 계산하여 최적의 Split을 찾는다. 

- XGBoost의 장점
    - Gradient Boosting에 비해 속도가 빠름
    - hyper-parameter를 조절할수 있음
    - 표준 GBM 경우 과적합 규제기능이 없으나, XGBoost는 자체적으로 규제기능이 있음
    - 분류와 회귀영역에서 뛰어난 예측 성능 발휘
  
## Ensemble Learning에서 Random_forest와 XG boost 실습코드

- ### 데이터 셋 소개
  
<p align="center"><<img width="357" alt="image" src="https://user-images.githubusercontent.com/97882448/204886022-b8e1104d-a86b-4958-bf5c-7eb1ca55d9be.png">
  
이번 데이터 셋은 차 구입에 대한 데이터로 변수도 성별, 나이,연간수익,차의 여부가 끝인 되게 간단한 데이터 셋이며 유저의 성별, 나이, 연간수익을 보고 자동차를 구입했는지 여부를 예측하는 분류문제이다. 
이전 데이터와 달리 이번 데이터는 상당히 간단한것을 알수있는데 선정이유는 2가지였다. 
  
첫번째는 과연 다른 지표에 비해 일상생활에서 쉽게 알수 있는 지표들(물론 연간수익을 공개하기는 쉽지 않은 문제일수도 있지만?)로 얼마나 예측을 정확히 할수 있을까?가 궁금했고 두번째는 살다보면 급하게 데이터를 처리해야 할일이 생기는데 순수한 궁금증으로 Auto ML로 알려진 pycaret과 비교했을때 hyper-parameter를 얼마나 잘 찾아주는지가 궁금하였다.
- ### 코드설명

```python
# 필요한 모듈 부르기
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```
필요한 모듈을 불러온다.
```python
df = pd.read_csv("car_data.csv")
#데이터의 머리부분 추출
df.head(5)
```
```python
df.info()
#데이터셋은 가장 간단한 UserID, 성별인 Gender, 나이, 연수입과 구입 여부(차를 구입하였으면 1, 아니면 0으로 정리하였음)
```
<p align="center"><img width="361" alt="image" src="https://user-images.githubusercontent.com/97882448/204887847-3a058982-1961-44d9-bcaf-b6b93012dc16.png">

데이터셋의 정보가 나오면서 UserID, 성별, 나이, 연수입과 차의 구입 여부(차를 구입하였으면 1, 아니면 0)의 대한 데이터셋이 구성됨

```python
#성별과 년수입의의 histplot을 본다. 
sns.histplot(x='AnnualSalary', data=df, hue='Gender', bins=20, multiple="dodge", shrink=.8)
```
<p align="center"><img width="500" alt="image" src="https://user-images.githubusercontent.com/97882448/204888370-d9a55d1f-821a-4e06-80cf-986254fafe72.png">

성별과 년수입의의 histplot을 본다. 남자보다 여성이 년수입이 더 많은 데이터라는것을 알수 있다. 

```python
#나이와 성별의 histplot을 본다.
sns.histplot(x='Age', data=df, hue='Gender', bins=20, multiple="dodge", shrink=.8)
```
<p align="center"><img width="500" alt="image" src="https://user-images.githubusercontent.com/97882448/204888975-b9dd2340-86e2-4267-9001-73a1d35a60aa.png">
  
```python
#차의 구입과 년수입의의 histplot을 본다.
sns.histplot(x='AnnualSalary', data=df, hue='Purchased', bins=20, multiple="dodge", shrink=.8)
```
<p align="center"><img width="500" alt="image" src="https://user-images.githubusercontent.com/97882448/204889130-1c8dac0a-bbb6-400b-a3ca-a3f62aa8b36d.png">

```python
plt.style.use("default")
sns.pairplot(df)
#데이터간의 선형관계가 있는지 확인해봄
```
<p align="center"><img width="529" alt="image" src="https://user-images.githubusercontent.com/97882448/204889341-acfbde3f-2b07-43a6-ac2c-fa9eebebef09.png">
  
pairplot을 확인해보며 데이터 간에 선형관계가 있는지 아무 관계가 없어 보이는지 시각적으로 감을 잡을수있다.

```python
#annot은 각 cell값의 값의 표기유무임 cmap은 Heatmap의 색을 표시해줌
sns.heatmap(df.corr(), annot=True, cmap='Blues',linewidths=0.5)
```
<p align="center"><img width="427" alt="image" src="https://user-images.githubusercontent.com/97882448/204889720-43309e91-ac2e-473b-ad5f-c4936270c81e.png">

annot은 각 cell안의 값의 표기유무이고 cmap은 Heatmap의 색을 표시해주는데 내용의 통일성을 위해 파란색의 heatmap으로 설정하였음. 데이터간의 배열을 색을 이용하여 표현한 그래프입니다.

```python
categorical = ["Gender"] # 위 데이터에서 gender만 categorical 데이터이다.
#categorical변수를 get_dummy를 통해 수치형변수로 바꿔준다.
df_final = pd.get_dummies(df, columns = categorical, drop_first = True)
df_final
```
<p align="center"><img width="314" alt="image" src="https://user-images.githubusercontent.com/97882448/204890824-ac5fcdd3-293c-40ba-8d35-59a6fd827cd7.png">
  
gender를 categorical로 설정 후 에 get_dummy를 통해 numurical로 바꾸줌 그럼 아래와 같이 데이터 프레임이 완성됨
  
```python
X = df_final.drop('Purchased', axis=1)
y = df_final['Purchased']
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.3, random_state= 42)
```
sklearn을 통해 train:test=7:3으로 나누어졌음

- #### Random forest
```python
#Random forest시작
rf = RandomForestClassifier(random_state = 31)
all_accuracies = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = 5)
print(all_accuracies)
#[0.94285714 0.85714286 0.85       0.88571429 0.94285714]
print (f"Mean of Accuracy : {all_accuracies.mean()}") # Average of all accuracies
print (f"Standard Deviation of Accuracy : {all_accuracies.std()}") # Standard Deviation of all accuracies
#Mean of Accuracy : 0.8957142857142857
#Standard Deviation of Accuracy : 0.040304959941902536
```
랜덤 포레스트를 돌리고 Randomstate는 31로 설정한후에 cross_validation은 5로 설정하였다. 결과는 #[0.94285714 0.85714286 0.85       0.88571429 0.94285714]로 나왔고 정확도 기준으로 정확도의 평균과 정확도의 표준편차를 구하였다. 이 지표는 나중에 Auto ML인 pycaret과 비교해볼것이다. 

```python
# 랜덤포레스트 나무의 수
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
# 모든 스플릿마다 변수의 갯수를 고려하는지
max_features = ["auto", "sqrt"]
# 나무의 최대 깊이
max_depth = [2, 4]
# 노드를 분할하는 데 필요한 최소 샘플 수
min_samples_split = [2, 5]
# 각 잎 노드에 필요한 최소 샘플 수
min_samples_leaf = [1, 2]
# bootstrap을 할건지 말건지
bootstrap = [True, False]
```
그리드 서치에 필요한 hyper-parameter의 범위를 설정해줌
```python
# 파라미터 그리드 생성
param_grid = {"n_estimators": n_estimators,
               "max_features": max_features,
               "max_depth": max_depth,
               "min_samples_split": min_samples_split,
               "min_samples_leaf": min_samples_leaf,
               "bootstrap": bootstrap}

print(param_grid)
#{'n_estimators': [10, 17, 25, 33, 41, 48, 56, 64, 72, 80], 'max_features': ['auto', 'sqrt'], 'max_depth': [2, 4], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2], 'bootstrap': [True, False]}
```
```python
from sklearn.model_selection import GridSearchCV
#그리드서치로 모델 튜닝이 진행 서치의 목적함수는 accuracy이고 n_jobs=-1은 모든 코어를 사용하겠다는 뜻임
rf_Grid = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, scoring = "accuracy", n_jobs = -1)
rf_Grid.fit(X_train, y_train)
```
그리드서치를 통해 모델 튜닝이 진행 서치의 목적함수는 accuracy이고 n_jobs=-1은 모든 코어를 사용하겠다는 뜻임
```python
  #최소의 파라미터 추출하기
rf_Grid.best_params_
```
<p align="center"><img width="300" alt="image" src="https://user-images.githubusercontent.com/97882448/204895466-983d2bea-b2ae-4b26-b170-563e5688c131.png">

```python
best_result = rf_Grid.best_score_
print(best_result)
#0.9042857142857142
```

그리드 서치를 진행한 결과 모델이 튜닝이 되어서 accuracy가 0.9042로 올라간것을 볼수 있음
```python
rf.feature_importances_
#첫번째부터 나열하면 나이,연수입,성별의 중요도
#array([0.48425947, 0.50592673, 0.0098138 ])
```
feature impotance는 나이보다 연수입이 차를 가지고 있는것에 더 중요한 부분을 차지함

- #### XGboost
```python 
#xgb
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
```
<p align="center"><img width="500" alt="image" src="https://user-images.githubusercontent.com/97882448/204897199-a893c274-3b4c-4132-8b29-5c0a3420cd68.png">

xgboost 모델정의하고 학습시키기
  
```python 
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy of the XG_boost_Model: ",accuracy)
#Accuracy of the XG_boost_Model:  88.66666666666667
```
XGboost의 정확도가 88.66정도나옴
```python 
param_grid = {
    'model__max_depth': [2, 3, 5, 7, 10],
    'model__n_estimators': [10, 100, 500],
}

grid = GridSearchCV(xgb_model, param_grid, cv = 5, n_jobs = -1, scoring = 'roc_auc')
grid.fit(X_train, y_train) 
```
RF와 마찬가지로 grid_search이지만 scoring 내가 관심있는 지표가 이번엔 roc_auc를 구하기로 하였음
  
```python 
print('best_param:', grid.best_params_)
print('best_score:', grid.best_score_)
#best_param: {'model__max_depth': 2, 'model__n_estimators': 10}
#best_score: 0.9546021731220089
```
그리드 서치를 진행한 결과 모델이 튜닝이 되어서 roc_auc가 0.954 올라간것을 볼수 있음
```python 
xgb_model.feature_importances_
#첫번째부터 나열하면 나이,연수입,성별의 중요도
#array([0.55401623, 0.33119002, 0.11479378]
```
RF와 달리 XGboost는 나이가 연수입보다 차를 가지고있는 것에 중요한 지표임
- #### pycaret_RF
```python
#pycaret은 파이썬 버전에 되게 민감함, 따라서 가상환경을 통해 버전을 3.8로 맞춰 준후 !pip install pycaret으로 설치해주어여함
from pycaret.classification import *
#import sys로는 현재 파이썬 버전을 알수 있음
import sys
print(sys.version)
```
pycaret은 파이썬 버전에 되게 민감함, 따라서 가상환경을 통해 버전을 3.8로 맞춘다음에 !pip install pycaret으로 설치해주어여합니다.
```python
clf = setup(data = df_final, train_size=0.7, target='Purchased',numeric_features=['Gender_Male','AnnualSalary','Age'],session_id = 31)
best=compare_models()
```
clf를 통해 trainsize를 맞추어주고 예측변수인 구입여부를 설정한후에 성별(위에서 코딩을 통해 숫자형으로 바꿔줌),연간수입,나이를 숫자형변수로 설정하였습니다. 후에 best=compare_models()을 통해 가능한 classification에서 best model을 찾으면 아래와 같이 그림이 나옵니다. 노란색은 가장 성능이 좋은것을 뜻합니다. 
<p align="center"><img width="500" alt="image" src="https://user-images.githubusercontent.com/97882448/204904405-71b46631-e8f2-476c-9959-a8d70a1a0039.png">

```python
rf = create_model('rf', fold=5)
```

우리가 비교하고자한것은 RF이니 fold를 5로 설정해서 성능을 확인해봅니다. 

<p align="center"><img width="400" alt="image" src="https://user-images.githubusercontent.com/97882448/204904688-a0912c2d-de5e-45fe-9920-a3adb6eeeb77.png">

randomforestclassfier()를 사용했을때보다  Mean of Accuracy : 0.8957142857142857 가 조금감소 되었고 표준편차는 Standard Deviation of Accuracy : 0.040304959941902536와 비교했을때 Auto ML이 더 작음을 알수 있습니다. 

```python
tune_rf=tune_model(rf,fold=5,optimize="Accuracy")
#그리드 서치를 한것처럼 Accuracy를 목적으로 tuning을 한번 시켜봄
```
<p align="center"><img width="400" alt="image" src="https://user-images.githubusercontent.com/97882448/204905815-bf8f896b-b238-4e98-afd0-98cf3d11da16.png">

Auto ML로 튜닝한것은 0.9084로 grid search한 0.9042보다 성능이 능가하진 못했음
```python
evaluate_model(tune_rf)
#마지막으로 모델을 평가하는 단계임 
```
<p align="center"><img width="800" alt="image" src="https://user-images.githubusercontent.com/97882448/204907314-19c43a9c-40b2-48b9-b645-372d04c8d400.png">

AUC말고도 시각적으로 feature importance등 다양한 평가지표를 알수 있음
- #### pycaret_XGboost
```python
xgboost = create_model('xgboost', fold=5)
```
<p align="center"><img width="400" alt="image" src="https://user-images.githubusercontent.com/97882448/204908152-10caaa89-26fb-4eb1-8013-0b09970e0ddc.png">

XGboost의 정확도가 88.66정도나오는데 Auto ML이 0.8884로 더 좋게나왔음 

```python
#xgbclassifier()처럼 AUC를 목적함수로 모델을 튜닝하였음
tune_xg=tune_model(xgboost,fold=5,optimize='AUC' )
```
<p align="center"><img width="400" alt="image" src="https://user-images.githubusercontent.com/97882448/204909568-ab4f2016-a798-4ece-a81a-5c97650562c4.png">

xgbclassifier()에서 튜닝한 auc의 best_score가 0.954가 나왔는데 0.957로 AutoML이 조금더 좋게나왔음 

```python
evaluate_model(tune_xg)
```
<p align="center"><img width="800" alt="image" src="https://user-images.githubusercontent.com/97882448/204909417-9e34b6b8-03ed-4e90-8bee-2c3e9c45e1b9.png">

AUC말고도 시각적으로 feature importance등 다양한 평가지표를 알수 있음

- ### 결론
 
sklearn과 AutoML에 관하여 모델의 기본 성능과 조그만한 하이퍼 파라미터 튜닝 했을때의 결과를 알고 싶었다. 예상하기에는 Auto ML(pycaret)이 sklearn기반이니 sklearn으로 RF와 XGboost를 예측한것과 별반차이가 없으나 AutoML이 더 좋지 않을것이다.라고 생각했지만 XGboost같은 경우에는  AutoML로 만든 XGboost가 성능이 더 좋았다. 그러니 데이터셋이 간단하고 전처리가 되어있는 깔끔한 classification문제이고 주어진 시간이 촉박한 경우는 Auto ML을 사용하는것이 괜찮은 방법같기도하다. 왜냐면 evaluation을 통해 시각적으로 훌륭한 자료를 많이 주기때문이다. 단, 연구를 할때에는 Auto ML은 어떤 모델이 성능이 좋게 나오는지 확인용 초반에만 사용하고 sklearn을 통한 앙상블 모델을 통해 세세히 하이퍼 파라미터를 바꾸어주는것이 좋겠다.  또한 RF와 XGboost를 통해 Feature importance를 뽑아 보았을때  RF일때는 차의 유무가 연간수입이 나이보다 중요도가 높았고 XGboost에서 차의 유무는 나이가 연간수입보다 중요도가 높았다. 성별은 RF와 XGboost 둘다 차의 유무에 중요한 변수가 되지 못했다. 
  
---
 ### Reference
 1. https://sustaining-starflower-aff.notion.site/2022-2-0e068bff3023401fa9fa13e96c0269d7 <강필성교수님 자료>
 2. https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset <CCars - Purchase Decision Dataset>
