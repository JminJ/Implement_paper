import torch
import torch.nn as nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, input, embedding_size):
        super(Embedding, self).__init__()
        # input은 bpe 알고리즘으로 전처리 및 padding 되어있을 것입니다.
        self.embedding_size = embedding_size # gpt-1 : 768

        self.embedding = nn.Embedding(self.input_len[-1], self.embedding_size) # self.input_len[-1] = (bs, seq, vocab_size)에서 vocab_size

    ### positional encodding~
    def cal_positional_encodding(self, position_now, i): # 현재 position과 i를 사용해 pos_encodding을 연산
        pos_encodding = position_now / torch.pow(10000, (2*(i//2))/self.embedding_size)
        return pos_encodding

    def positional_encodding(self, position_now): # 현재 position 값과 embedding_size(768)를 사용해 cal_positoinal_encodding을 호출
        return [cal_positional_encodding(powition_now, i) for i in range(self.embedding_size)]

    def get_sinusoiding_table(self, input_len): # 연산된 pos_encodding에 i값에 따라 sin, cos을 적용
        sinusoid_table = np.array([positional_encodding(i_seq) for i_seq in range(input_len[1])])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return sinusoid_table # 이것이 positional encodding 결과값
    ### ~positional encodding

    ### embedding~
    def forward(self, input):
        # input을 forward에서 받아오고 input_len을 get_sinusoding_table 함수로 보냄
        input_len = input.size()
        embedding_input = self.embedding(input) / self.embedding_size**0.5 # embedding_size(d_model)의 루트로 나눠준다.
        pos_embedding = get_sinusoiding_table(input_len)

        return embedding_input + pos_embedding
    ### ~embedding
        
        
class self_dot_attention(nn.Module):
    def __init__(self, Q, K, V): # multi-head-attention에서 나눈 Q, K, V
        super(self_dot_attention, self).__init__()
        self.Q = Q
        self.K = K
        self.V = V
        self.softmax = nn.Softmax()

    def forward(self, attn_mask):
        # Q와 K를 matmul하고 scale을 한다.
        matmul = torch.matmul(self.Q, torch.transpose(K,-1, -2)) / self.K.size(-1) ** 2
        print(matmul.size())

        # mask를 추가한 값을 softmax에 넣고 V와 곱해준다.
        soft_mat = self.softmax(matmul, dim=-1)
        mul_v = torch.matmul(soft_mat, self.V)

        return mul_v
    

class multi_head_attention(nn.Module):
    def __init__(self, Q, K, V, n_head, batch_size, d_model):
        super(multi_head_attention ,self).__init__()
        self.Q = Q
        self.K = K
        self.V = V
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model / n_head

        self.W_Q = nn.Linear(self.Q.size(-1), n_head * (d_model/n_head))
        self.W_K = nn.Linear(self.K.size(-1), n_head * (d_model/n_head))
        self.W_V = nn.Linear(self.V.size(-1), n_head * (d_model/n_head))

        self.last_linear = nn.Linear(self.model, self.d_model)

    def forward(self):
        W_Q = self.W_Q(self.Q).view(self.batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        W_K = self.W_K(self.K).view(self.batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        W_V = self.W_V(self.V).view(self.batch_size, -1, self.n_head, self.d_head).transpose(1,2)

        self_attention = self_dot_attention(W_Q, W_K, W_V)
        print(self_attention.size())

        concat_attentions = self_attention.contiguous().view(self.batch_size, -1, self.d_model)

        output = self.last_linear(concat_attentions)

        return output

