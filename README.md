RNN GRU LSTM
=> 제일 초기 모델

LSTM_2
- lr 감소 확인 추가
- loss 항목별로 추가
- weight 추가
  weight 설정1 (w1)
SIGN_W = 0.5
O3_W   = 3.0
O2_W   = 1.0
O1_W   = 1.0
O0_W   = 2.0
  weight 설정2 (w2)
SIGN_W = 0.5
O3_W   = 3.0
O2_W   = 1.2
O1_W   = 1.5
O0_W   = 2.5

LSTM3
- orbd 3개로 줄인 버전, weight=1
- (1M 기준으로 탐색 결과 LBA 최대 11자리, 4096 으로 나눠놓은 것을 탐색하니까 7자리 => orbd 3개)

LSTM4
- alibaba_trace 전체를 확인했을 때 최대 LBA는 13자리 => 최댓값 10^14를 4096으로 나누면 최대 2.44*10^10
- orbd 4개 각 512 weight=1

LSTM 5
- LSTM4 에 weight 추가 버전
- orbd 512
- weight
SIGN_W = 3.0
O3_W   = 3.0
O2_W   = 2.0
O1_W   = 1.0
O0_W   = 0.5

LSTM6
- LSTM5 에서 decay의 epoch를 0.9=>0.8 로 수정
- 결과: 다소 나아지긴 했으나 큰 차이는 없음.
- lr2~5 까지 각 loss 5 4.6 4.9 6.1