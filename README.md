# BA_04 Ensemble Learning(Bagging & Boosting)

<p align="center"><img width="604" alt="image" src="https://user-images.githubusercontent.com/97882448/204542304-85e7f83d-075c-4d02-b5e8-6e8d77d66987.png">

이자료는 고려대학교 비니지스 애널리틱스 강필성교수님께 배운 Ensemble Learning(Bagging & Boosting)을 바탕으로 만들어졌습니다.
먼저, Ensemble Learning(Bagging & Boosting)방식의 기본적인 개념을 배운후에 Bagging에선 Random Forest,Boosting에선 XG Boost에 대해 개념 및 코드를 통해 직접 구현을 해봄으로써 이해를 돕도록하겠습니다. 또한 Auto ML과의 성능차이 및 시간차이를 통해 간단한 실험을 해보겠습니다.

## BA_04 Ensemble Learning_개념설명
Ensemble Learning 이란 여러 개의 분류기(Classifier)를 생성하고 그 예측을 결합함으로써 보다 정확한 예측을 도출하는 기법을 말합니다. 그러면 왜? 이 앙상블 러닝을 사용하는지 궁금하실수 있습니다.  이유는 먼저 하나의 Single 모델이 모든 데이터셋에서 우수하지 않습니다.  그래서 여러 다양한 학습 알고리즘들을 결합하여 학습시키는 방법을 고안하였습니다. 따라서 예측력의 보완되고  각각의 알고리즘을 single로 사용할 경우 나타나는 단점들을 보완되어집니다. 쉽게 말해서 옛 속담에 백지장도 맞들면 낫다와 같이 집단 지성을 이용하는 방법입니다.
