# BA_04 Ensemble Learning(Bagging & Boosting)

<p align="center"><img width="604" alt="image" src="https://user-images.githubusercontent.com/97882448/204542304-85e7f83d-075c-4d02-b5e8-6e8d77d66987.png">

이자료는 고려대학교 비니지스 애널리틱스 강필성교수님께 배운 Ensemble Learning(Bagging & Boosting)을 바탕으로 만들어졌습니다.
먼저, Ensemble Learning(Bagging & Boosting)방식의 기본적인 개념을 배운후에 Bagging에선 Random Forest,Boosting에선 XG Boost에 대해 개념 및 코드를 통해 직접 구현을 해봄으로써 이해를 돕도록하겠습니다. 또한 Auto ML과의 성능차이 및 시간차이를 통해 간단한 실험을 해보겠습니다.

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
3사분면은 편향은 높지만 분산은 낮은 경우를 말합니다. 예를 들어 logistic regression이나  K-nn에서 K가 클 때를 말합니다. 2사분면은 편향은 낮지만 분산은 높은 경우립니다. 예를 들어 Decision Tree나  K-nn에서 K가 작을 때를 말합니다. 이토록 현실에선 3사분면과 2사분면 같은 모델이 대부분입니다. 따라서 1사분면같이 편향과 분산이 둘 다 낮아야 좋은 모델을 충족시키기 위해 Ensemble방식이 제안되어집니다. 2사분면의 분산을 줄여줘서 1사분면처럼 만드는 방식이 Bagging이고 3사분면의 편향을 줄여줘서 1사분면처럼 만드는 방식이 Boosting입니다. 
