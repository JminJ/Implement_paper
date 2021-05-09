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


class masking():
    def __init__(self, input, Q_size, K_size):
        self.input = input # input은 rnn_sequence_pad를 통해 패딩 된 상태의 값입니다.
        self.Q_size = Q_size
        self.K_size = K_size

    def forward(self):
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
        super(self_dot_attention, self).__init__()
        self.Q = Q
        self.K = K
        self.V = V
        self.softmax = nn.Softmax()

    def forward(self, attn_mask):
        # Q와 K를 matmul하고 scale을 한다.
        matmul = torch.matmul(Q, torch.transpose(K,-1, -2)) / self.K.size(-1) ** 2
        print(matmul.size())

        # masking을 한다.
        matmul.masked_fill_(attn_mask, '-inf')

        # mask를 추가한 값을 softmax에 넣고 V와 곱해준다.
        soft_mat = self.softmax(matmul, dim=-1)
        mul_v = torch.matmul(soft_mat, self.V)

        return mul_v


class masked_multi_head_attention(nn.Module):
    def __init__(self, Q, K, V, n_head, batch_size, d_model): # d_model을 빼고 Q.size(-1)을 쓸까 고민...
        super(multi_head_attention, self).__init__()
        self.Q = Q
        self.K = K
        self.V = V
        self.n_head = n_head # gpt-1 : 12
        self.d_model = d_model
        # self.d_head = self.Q.size(-1) / n_head
        self.d_head = self.d_model / n_head # 768(d_model = Q.size(-1) = embedding_size) / 12
        self.batch_size = batch_size
        
        self.linear_WO = nn.Linear(self.d_model, self.d_model)

    def forward(self, input):
        Q_head = nn.Linear(self.Q.size(-1), self.n_head * self.d_head) # (bs, seq, d_model)
        K_head = nn.Linear(self.K.size(-1), self.n_head * self.d_head)
        V_head = nn.Linear(self.V.size(-1), self.n_head * self.d_head)

        Q_head = Q_head.view(self.batch_size, -1, self.n_head, self.d_head).transpose(1, 2) # (bs, seq, n_head, d_head) => (bs, n_head, seq, d_head)
        K_head = K_head.view(self.batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        V_head = V_head.view(self.batch_size, -1, self.n_head, self.d_head).transpose(1, 2)


        attn_mask = masking(input, Q_head.size(), K_head.size()) # masked-multi-head attention을 위해 triu를 적용한 masking 사용
        multi_attn_mask = attn_mask(input, self.Q.size(), self.K.size()).unsqueeze(1).repeat(1, self.n_head, 1, 1) # multi-head 사이즈에 맞게 (Q.size(1)(bs), n_head(12), Q.size(1), K.size(1))로 바꾼다
        
        # multi_attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1) # multi-head 사이즈에 맞게 (Q.size(1)(bs), n_head(12), Q.size(1), K.size(1))로 바꾼다.

        self_dot_attn = self_dot_attention(Q_head, K_head, V_head) # self-dot-attention
        multi_attn_result = self_dot_attn(multi_attn_mask)

        concat_result = multi_attn_result.contiguous().view(self.batch_size, -1, self.d_model) # self-dot-attention의 결과를 concatenation한다.

        result = self.linear_WO(concat_result)

        return result
        
        
class position_wise_FFN(nn.Module):
    def __init__(self, d_model):
        super(position_wise_FFNN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4) # transformer에서 4배를 해줌으로 4배로 지정해 보았습니다 :)
        self.GELU = nn.GELU()
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input_x): # input_x = multi-head-attention 결과값(bs, seq, d_model)
        linear_1st = self.linear1(input_x) # (bs, seq, d_model * 4)
        gelu_result = self.GELU(linear_fir) 
        linear_2nd = self.linear2(gelu_result) # (bs, seq, d_model) 

        return linear_2nd


class transformer_block(nn.Module): # transformer의 decoder(n_laryer = 12, d_model = 768, self_attn_head = 12)
    def __init__(self, input, vocab_size, n_layer, d_model, self_attn_head = 12): ### encodding 함수에서 vocab_size를 output으로 내놓아야 할 것 ###
        super(transformer_block, self).__init__()
        self.input = input
        self.n_layer = n_layer
        self.d_model = d_model
        self.self_attn_head = self_attn_head

        self.result = None
        
        self.masked_multi_head_attn = masked_multi_head_attention(input, input, input, n_layer, d_model, self_attn_head)
        self.position_wise = position_wise_FFN(d_model)

        self.dropout = nn.Dropout(0.1) # 각 sub-layer(Add & Norm되기 전 작업들)마다 적용해 줄 dropout

    def add_and_normalization(self, before_result, new_result):
        new_result = before_result + now_result

        layer_norm = nn.LayerNorm(new_result.size())
        layer_norm_result = layer_norm(new_result)

        return layer_norm_result

    def block(self):
        masked_result = self.masked_multi_head_attn(self.input)
        masked_result = add_and_normalization(self.input, masked_result)

        feed_for_result = self.position_wise(masked_result)
        feed_for_result = add_and_normalization(masked_result, feed_for_result)

        return feed_for_result

    def forward(self):
        for _ in range(self.n_layer):
            self.result = block(self.input) # n_layer의 수 만큼 block을 실행
            self.input = self.result # self.result가 self.input으로 block의 입력으로 들어간다.

        return self.result


class GPT_pretrain(nn.Module):
    def __init__(self, vocab_size, n_layer, d_model, self_attn_head):
        self.embedding = Embedding(input_size, embedding_size)
        self.transformer_block = transformer_block(embedding, vocab_size, n_layer, d_model, self_attn_head) # sentencepiece -> dataloader -> Embedding() -> trans?
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, input):
        embedding_input = self.embedding(input)
        
        self.linear.weight = self.embedding.embedding.weight

        result_tran = self.transformer_block()
        result_linear = self.linear(result_tran)

        return result_linear[:, :-1, :].contiguous()