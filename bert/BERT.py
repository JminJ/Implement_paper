import torch
import torch.nn as nn
import numpy as np
        
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

        return mul_v, soft_mat
    

class multi_head_attention(nn.Module):
    def __init__(self, Q, K, V, n_head, batch_size, d_model):
        super(multi_head_attention, self).__init__()
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

        self_attention, soft_mat = self_dot_attention(W_Q, W_K, W_V)
        print(self_attention.size())

        concat_attentions = self_attention.contiguous().view(self.batch_size, -1, self.d_model)

        output = self.last_linear(concat_attentions)

        return output, soft_mat

class Encoder_Layer(nn.Module):
    def __init__(self, inputs, n_head, batch_size, d_model, seq_len_size):
        super(Encoder_Layer, self).__init__()
        self.inputs = inputs
        self.n_head = n_head
        self.batch_size = batch_size
        self.d_model = d_model
        self.seq_len_size = seq_len_size

        self.m_Q = nn.Linear(self.inputs.size(-1), d_model)
        self.m_K = nn.Linear(self.inputs.size(-1), d_model)
        self.m_V = nn.Linear(self.inputs.size(-1), d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)

        self.ffnn_linear1 = nn.Linear(d_model * 4, d_model) # transformer에서 * 4를 취해주므로 본 코드에서도 따라갑니다.
        self.ffnn_gelu = nn.GeLU() # ReLU 대신 GeLU를 사용합니다.
        self.ffnn_linear2 = nn.Linear(d_model * 4, d_model) # transformer에서 * 4를 취해주므로 본 코드에서도 따라갑니다.

    def first_hidden_state(self):
        initial_hidden = torch.random(self.batch_size, self.seq_len_size, self.d_model)

        return initial_hidden

    def add_and_norm(self, before_result, now_result):
        add_result = before_result + now_result
        norm_result = self.layer_norm(add_result)
        
        return norm_result

    def feed_forward_function(self, result):
        linear1 = self.ffnn_linear1(result)
        gelu_result = self.ffnn_gelu(linear1)
        linear2 = self.ffnn_linear2(gelu_result)

        return linear2

    def forward(self, Q, K, V):
        before_result = self.first_hidden_state()
        attn_result, soft_mat = multi_head_attention(Q, K, V, self.n_head, self.batch_size, self.d_model)

        a_n_result = self.add_and_norm(before_result, attn_result)
        
        feed_result = self.feed_forward_function(a_n_result)

        return feed_result, soft_mat

class Encoder(nn.Module):
    def __init__(self, inputs, n_head, batch_size, d_model, seq_len_size):
        super(Encoder, self).__init__()
        self.inputs = inputs
        self.n_head = n_head
        self.batch_size = batch_size
        self.d_model = d_model
        self.seq_len_size = seq_len_size

        self.encoder_layers = Encoder_Layer(inputs, n_head, batch_size, d_model, seq_len_size)
        
    def forward(self):
        soft_mats = []

        result, soft_mat = self.encoder_layers(self.inputs)
        for _ in range(self.n_head-1):
            result, soft_mat = self.encoder_layers(result)
            soft_mats.append(soft_mat)
        return result, soft_mats

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        pass


        

    

