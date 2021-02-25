import torch
import torch.nn as nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, input, input_size, embedding_size):
        # input은 bpe 알고리즘으로 전처리 및 0으로 padding 되어있을 것입니다.
        self.input = input
        self.input_len = input.size()
        self.input_size = input_size
        self.embedding_size = embedding_size # gpt-1 : 768

        self.embedding = nn.Embedding(self.input_size, self.embedding_size)

    ### positional encodding~
    def cal_positional_encodding(self, position_now, i): # 현재 position과 i를 사용해 pos_encodding을 연산
        pos_encodding = position_now / torch.pow(10000, (2*(i//2))/self.embedding_size)
        return pos_encodding

    def positional_encodding(self, position_now): # 현재 position 값과 embedding_size(768)를 사용해 cal_positoinal_encodding을 호출
        return [cal_positional_encodding(powition_now, i) for i in range(self.embedding_size)]

    def get_sinusoiding_table(self): # 연산된 pos_encodding에 i값에 따라 sin, cos을 적용
        sinusoid_table = np.array([positional_encodding(i_seq) for i_seq in range(self.input_len[1])])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return sinusoid_table # 이것이 positional encodding 결과값
    ### ~positional encodding

    ### embedding~
    def cal_embedding(self):
        embedding_input = self.embedding(self.input) / self.embedding_size**0.5 # embedding_size(d_model)의 루트로 나눠준다.
        pos_embedding = get_sinusoiding_table()

        return embedding_input + pos_embedding
    ### ~embedding

class masking():
    def __init__(self, input, Q_size, K_size):
        self.input = input # input은 rnn_sequence_pad를 통해 패딩 된 상태의 값입니다.
        self.Q_size = Q_size
        self.K_size = K_size

    def fill_mask(self):
        # match_sized_input = torch.masked_fill(self.input)
        input_zero = self.input.eq(0).unsqueeze(1).expand(self.Q_size[0], self.Q_size[1], self.K_size[1])
        print('|input_zero| :', input_zero.size())

        mask_vec = torch.ones_like(self.input).expand(self.Q_size[0], self.Q_size[1], self.K_size[1])
        mask_vec = torch.triu(mask_vec)

        plus_result = torch.gt((input_zero + mask_vec), 0) # triu된 값과 input에서 패딩(0)된 부분을 torch.gt를 통해 합친다.
        # print(plus_result[1])

        return plus_result
        
class self_dot_attention(nn.Module):
    def __init__(self, Q, K, V): # multi-head-attention에서 나눈 Q, K, V
        self.Q = Q
        self.K = K
        self.V = V
        self.softmax = nn.Softmax()

    def forward(self, attn_mask):
        # Q와 K를 matmul하고 scale을 한다.
        matmul = torch.matmul(Q, torch.transpose(K,-1, -2)) / self.K.size(-1) ** 2
        print(matmul.size())

        # masking을 한다.
        matmul.masked_fill(attn_mask, -1e9)

        # mask를 추가한 값을 softmax에 넣고 V와 곱해준다.
        soft_mat = self.softmax(matmul, dim=-1)
        mul_v = torch.matmul(soft_mat, self.V)

        return mul_v

class multi_head_attention(nn.Module):
    def __init__(self, Q, K, V, n_head, batch_size):
        self.Q = Q
        self.K = K
        self.V = V
        self.n_head = n_head # gpt-1 : 12
        self.d_head = Q.size(-1) / n_head # 768(d_model = Q.size(-1) = embedding_size) / 12
        self.batch_size = batch_size

    def cal_multihead_attention(self, input):
        Q_head = nn.Linear(self.Q.size(-1), self.n_head * self.head_linear) # (bs, seq, d_model)
        K_head = nn.Linear(self.K.size(-1), self.n_head * self.head_linear)
        V_head = nn.Linear(self.V.size(-1), self.n_head * self.head_linear)

        Q_head = Q_head.view(self.batch_size, -1, self.n_head, self.d_head).transpose(1, 2) # (bs, seq, n_head, d_head) => (bs, n_head, seq, d_head)
        K_head = K_head.view(self.batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        V_head = V_head.view(self.batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        attn_mask = input.eq(0).unsqueeze(1).expand(self.Q.size(0), self.Q.size(1), self.K.size(1))
        multi_attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        self_dot_attn = self_dot_attention(Q_head, K_head, V_head)
        multi_attn_result = self_dot_attn(attn_mask)

        concat_result = nn.Linear(self.d_head, self.n_head * self.d_head)

        