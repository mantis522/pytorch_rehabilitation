import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import random
import pandas as pd
import re
from torchtext.legacy.data import TabularDataset
from torchtext.legacy import data
from sklearn.model_selection import train_test_split

SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)

# 하이퍼파라미터
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 3

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

 # torchtext.data 임포트

# 필드 정의
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

train_data, test_data = TabularDataset.splits(
        path=r"D:\ruin\data\csv_file\imdb_split", train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))

TEXT.build_vocab(train_data, min_freq=5) # 단어 집합 생성
LABEL.build_vocab(train_data)

n_classes = 2

vocab_size = len(TEXT.vocab)

train_data, val_data = train_data.split(split_ratio=0.8)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), sort=False,batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)



class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(DEVICE) # 첫번째 히든 스테이트를 0벡터로 초기화
        # h_0 = self._init_state(batch_size=x.size(0)) # 첫번째 히든 스테이트를 0벡터로 초기화
        x, _ = self.gru(x, h_0)  # GRU의 리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        h_t = x[:,-1,:] # (배치 크기, 은닉 상태의 크기(입력벡터 차원수))의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        ## [:, -1, :]는 첫번째는 미니배치 크기, 두번째는 마지막 은닉상태, 세번째는 모든 hidden_dim을 의미하는 것

        self.dropout(h_t)
        logit = self.out(h_t)  # (배치 크기, 은닉 상태의 크기) -> (배치 크기, 출력층의 크기)
        return logit

    # def _init_state(self, batch_size=1):
    #     weight = next(self.parameters()).data
    #     return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

model = GRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train(model, optimizer, train_iter):
    model.train() ## model 학습
    for b, batch in enumerate(train_iter):
        x, y = batch.review.to(DEVICE), batch.sentiment.to(DEVICE) ## 각각 x, y에 리뷰 데이터랑 라벨 넣기
        # y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()  # 갱신할 변수들에 대한 모든 변화도를 0으로 만듭니다. 이렇게 하는 이유는
        # 기본적으로 .backward()를 호출할 때마다 변화도가 버퍼(buffer)에 (덮어쓰지 않고) 누적되기 때문입니다.

        logit = model(x) ## 모델에 x를 넣고
        loss = F.cross_entropy(logit, y) ## 크로스 엔트로피 함수로 loss 함수 구함
        loss.backward() ## 역전파로 계산하면서
        optimizer.step() ## 매개변수 업데이트

def evaluate(model, val_iter):
    """evaluate model"""
    model.eval() ## eval은 그냥 함순가 봄
    corrects, total_loss = 0, 0 ## total_loss는 대충 알겠는데 corrects는 뭔지 모르겠네
    for batch in val_iter:
        x, y = batch.review.to(DEVICE), batch.sentiment.to(DEVICE)
        # y.data.sub_(1) # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

best_val_loss = None
for e in range(1, EPOCHS+1):

    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        # if not os.path.isdir("snapshot"):
        #     os.makedirs("snapshot")
        # torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss

# model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))