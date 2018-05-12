

### 목적의 중요성

-  Project를 진행할 때, 목적을 명확히 하지 않았다. Kaggle은 상위 5개의 class를 맞추는 것이었고, Project는 데이터의 인사이트를 도출 목적 가 명확하지 않았다.
-  그래서 Project를 진행할 때, 데이터를 



### 신뢰성의 중요성

- 데이터 분석의 목적은 '사장에게 자신이 만든 인사이트를 납득' 시키는 것이다. 그렇기 때문에 인사이트를 발표할 때 신뢰성을 주는 것이  중요!!
- 신뢰성을 높이기 위해, `논리적인 전개 과정`과 `Code 공유` 한다. 또한 Project를 진행할 때, 사장의 질문에 대해 끊임 없는 답변을 생각해 두어야 한다.



### Create features의 중요성

-  분별을 할 때, 제일 중요한 것은 분별할 수 있는 feature를 생성하는 게 중요
-  classification은 여러 모델이 존재하기 때문에 어떤 모델을 쓰는 것에 집중하게 된다. 하지만 실제 Perfomance에서는 어느 모델을 쓰는 것 보다는 class를 분별해 주는 feature를 생성할 때 성능을 많이 올리게 되었다.



### Create features할 때 생각해야 되는 것

- feature를 만든 이유를 설명할 때, feature의 EDA를 통해 발견한 insight로 설명해 주는 것이 좋다.



### 데이터 시각화에 대한 방법

- 실제 데이터를 가지고 시각화를 할 때, 전체 데이터를 보여준면 큰 차이가 보이지 않는 경우가 발생한다. 그렇기 때문에 mean, median으로 했다. 하지만 **log**를 취해 그 차이를 확대 시킬 수 있다.



### log loss에 대해서 ...

- log loss를 사용할 때, CV를 사용할 때에는 음수가 되고, error function으로 사용할 때에는 양수이다. 그 이유는 error는 적을 수록 좋고 CV는 높은 수록 좋게 만들기 위해서 CV에 음수를 사용했다.



### 비대칭적인 문제

- 일반적인 classification을 할 때, class가 한 쪽에 몰려 있으면 많은 데이터에 accurancy는 높지만 recall이 매우 낮게 나오는 경우가 발생하게 된다. 그 이유는 대부분 예측 값을 몰려 있는 데이터로 선택할 때, accurancy가 높게 나오기 때문이다.
- airbnb 같은 경우도 NDF에 값이 몰려 있기 때문에, 예측 값이 대부분이 NDF가 선택 된다.
- 이 문제를 해결하기 위해서는 over-sampling, under-sampling을 사용한다. 그 외에도 weight-class를 주어서 대칭적인 문제를 해결한다.
- 하지만 airbnb 문제에는 weight-class를 이용했을 때 recall은 높일 수 있지만 kaggle 점수는 낮아지기 때문에 사용하지 않았다.



- 적은 애들의 인사이트를 발견하기 위해서는 많이 있는 class를 제외하고 classification을 진행해 인사이트를 얻는다.  



### EDA의 중요성

- regression에도 느낀 것이지만 사용하는 feature들에 대해서 아는 것이 매우 중요하다. 
- EDA를 통해서 feature 생성에 대한 인사이트를 얻을 수 있고, 다른 사람들에게 설명을 할 때에도 큰 도움이 될 수 있다.
- But, EDA는 결론에 큰 도움이 될 수 없다. EDA로 힌트를 얻을 수 있지만, 결론을 도출하기에는 도움이 않된다.



### feature가 유의한지 않하는 가?

- binary 문제에 대한 유의성은 bernoulli test / One-way ANOVA로 판단할 수 있다.
- multiclass 문제에 대한 유의성은 One-way ANOVA로 판단할 수 있다.



### Ensemble 방법에 대한 Tree 분석

- Ensemble에 대한 Tree에 대한 분석은 큰 의미가 없다. 그 이유는 Ensemble에 사용되는 Tree는 weak classifier이기 때문에 분석이 큰 의미를 가지지 않는다. 