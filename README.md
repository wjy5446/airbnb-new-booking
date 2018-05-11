# kaggle : Airbnb New User Bookings

### 팀프로젝트

- 박성호, 윤재영, 조준기

### 개요

- Airbnb의 사용자가 어느나라를 예약할지 분류 예측문제
- NDCG 를 통한 분류모델 평가
- Light GBM 모델 사용

“최종 결과 : 0.88461 (상위 11% )”

### 데이터

Train.csv
- 트레이닝 데이터 : ​ (213451, 15)
- Target 데이터 : country_destination (12개 클래스)

Test.csv
- 테스트 데이터 : (62096, 14)
- 트레이닝 데이터를 통한 Target 데이터 예측

Sessions.csv
- 사용자 로그데이터: (10567737, 6)
- 사용자의 action, action type, action details, device type, session elapsed

Countries.csv (target값에 대한 정보로 최종 미사용)
- 목적지의 나라정보 : (10, 7)

Age_gender_bkts.csv (target값에 대한 정보로 최종 미사용)
- 목적지의 인구통계자료 : (420, 5)

### 데이터 엔지니어링

Null 데이터 비율
- Date_first_b​ ooking : 67% (테스트 데이터중 100%) (최종 미사용)
- Age : 42% (null 데이터에 한해 기존의 나이분류중 5가지 계층으로 null 데이터 채움)
- First affiliate tracked : 2% (mode 값으로 데이터 채움)

성별 데이터
- 남자, 여자, Unknown, others존재
- 교차검증기준 남자, 여자 데이터만 사용
- Light GBM을 통한 unknown, other를 남자, 여자로 분류

First active date / account create date
- 년, 월, 일 사용
- First active date를 기준으로 주말, 공휴일 데이터 사용

Lagging time : First active date 와 create account date의 시간차를 사용

Faithless sign-in
- Gender가 unkwon으로 입력 및 나이데이터를 입력하지 않은 경우의 데이터 사용

세션 데이터
- Action, action type : 사용자의 행동 및 어떤종류의 행동인지에 대한 데이터
- Count, mode데이터를 사용
- Session elapsed : 사용자가 채류한 시간데이터
- Mode, median, mean 데이터를 사용
- Device type : 어떤 종류의 장치인지에 대한 데이터
- Mode 데이터 사용

카테고리 데이터
- 카테고리 변수를 더미변수로 변경

“최종 1054개 Feature 사용”

### 모델링

교차검증

- NDCG의 경우 정답인 순서를 학습시켜야 해서 사용불가
- Log loss로 분류검증
- 로지스틱 회귀분석, SVM, Randomforest, Light GBM, Ensemble model등을 활용한 분류성능 평가

“교차검증기준 Light GBM이 구동시간 및 성능이 가장 우수”

- Light GBM 파라미터 튜닝 (교차검증기준 가장 성능이 우수한 parameter 선택)
Boost-type = ‘gbtr’
Evaluation metric = log loss
Learning rate = 0.1
Number of estimator = 100
L1 regularization = 1
L2 regularization = 0
모델 accuracy : 0.65, precision :0.69, recall : 0.66

### 결론
- NDCG 결과: 0.88461 (상위 11%)
"Target 데이터 imbalanced현상으로 특정 데이터의 recall이 높은데 One vs
One문제 및 stacking을 통한 성능개선 요지"
