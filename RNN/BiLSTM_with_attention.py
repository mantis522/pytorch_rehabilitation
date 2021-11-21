import os
import random
import torch
import torch.nn  as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchtext.legacy.data import TabularDataset
from torchtext.legacy import data
from tqdm import tqdm
from torchtext.vocab import Vectors
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

RANDOM_SEED = 2020
MAX_VOCAB_SIZE = 25000  ## 상위 25k개 단어만 사전에 넣겠다는 의미.
BATCH_SIZE = 128

torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

# TEXT = data.Field(tokenize='spacy', fix_length=1000)
TEXT = data.Field(tokenize=str.split, include_lengths=True)
LABEL = data.LabelField(dtype = torch.float)

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))
        # embedded = [sent len, batch size, emb dim]
        text_lengths = text_lengths.cpu()
        # text_length = [batch_size]
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                            text_lengths, enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        # hidden = self.dropout(torch.add(hidden[-2,:,:], hidden[-1,:,:]))
        # hidden = [batch size, hid dim]
        output = output.permute(1, 0, 2)
        # output = [batch size, sent len, hid dim *2]
        output = torch.tanh(output)

        attention = F.softmax(self.attn(output), dim=1)
        # print("attention shape is ", attention.shape)
        # attention = [batch size, sent len,1]
        attention = attention.squeeze(2)
        # attention = [batch size, sent len]
        attention = attention.unsqueeze(1)
        # attention = [batch size, 1, src len]
        representation = torch.bmm(attention, output)
        # representation = [batch size, 1 ,hid dim *2]
        representation = representation.squeeze(1)
        # representation = [batch size, hid_dim *2]
        representation = self.fc(representation)
        # representation = [batch size, output_dim]
        return representation, attention

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    # print("round preds ", len(rounded_preds))
    correct = (rounded_preds == y).float() #convert into float for division
    # print("correct is ", correct)
    acc = correct.sum() / len(correct)
    # print("acc is ", acc)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    pbar = tqdm(iterator, position=0, leave=True)

    for batch in pbar:
        optimizer.zero_grad()
        text, text_lengths = batch.review
        predictions, _ = model(text, text_lengths)
        predictions = predictions.squeeze(1)
        ## [128, 1]로 출력되는 것을 [128]로 만들어주기 위해 squeeze(1)을 넣어줌.
        loss = criterion(predictions, batch.sentiment)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        batch_acc = binary_accuracy(predictions, batch.sentiment)

        epoch_loss += batch_loss
        epoch_acc += batch_acc.item()

        pbar.set_description('train => acc {} loss {}'.format(batch_acc, batch_loss))

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        pbar = tqdm(iterator, position=0, leave=True)
        for batch in iterator:
            text, text_lengths = batch.review
            predictions, _ = model(text, text_lengths)
            predictions = predictions.squeeze(1)
            loss = criterion(predictions, batch.sentiment)
            batch_acc = binary_accuracy(predictions, batch.sentiment)
            batch_loss = loss.item()
            epoch_loss += batch_loss
            epoch_acc += batch_acc.item()

            pbar.set_description('eval() => acc {} loss {}'.format(batch_acc, batch_loss))

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def main():
    train_data, test_data = TabularDataset.splits(
        path=r"D:\ruin\data\csv_file\imdb_split", train='train_data.csv',
        test='test_data.csv', format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

    train_data, eval_data = train_data.split(random_state=random.seed(RANDOM_SEED))
    glove_vectors = Vectors(r"D:\ruin\data\glove.6B\glove.6B.100d.txt")

    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors=glove_vectors,
                     min_freq=10)
    LABEL.build_vocab(train_data)

    train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE,
                                     device=device,
                                     shuffle=True)
    eval_iter, test_iter = data.BucketIterator.splits((eval_data, test_data),
                                                      batch_size=BATCH_SIZE,
                                                      device=device,
                                                      sort_key=lambda x: len(x.review),
                                                      sort_within_batch=True)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = RNN(INPUT_DIM,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT,
                PAD_IDX)

    pretrained_embeddings = TEXT.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 10

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        print('{}/{}'.format(epoch, N_EPOCHS))

        train_loss, train_acc = train(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, eval_iter, criterion)

        print()
        print('Train => acc {:.3f}, loss {:4f}'.format(train_acc, train_loss))
        print('Valid => acc {:.3f}, loss {:4f}'.format(valid_acc, valid_loss))
        scheduler.step()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

    test_loss, test_acc = evaluate(model, test_iter, criterion)

    print()
    print('Test Eval => acc {:.3f}, loss {:4f}'.format(test_acc, test_loss))

if __name__ == "__main__":
    main()