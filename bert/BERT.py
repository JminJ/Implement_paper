import torch
import torch.nn as nn
import numpy as np
        
def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad)
    pad_attn_mask= pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask

class self_dot_attention(nn.Module):
    def __init__(self): # multi-head-attention에서 나눈 Q, K, V
        super(self_dot_attention, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, Q, K, V, attn_mask):
        # Q와 K를 matmul하고 scale을 한다.
        matmul = torch.matmul(self.Q, torch.transpose(K,-1, -2)) / self.K.size(-1) ** 2
        print(matmul.size())
        # seq 길이 pad mask 적용
        matmul.masked_fill_(attn_mask, -1e9)

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

        self.self_attn = self_dot_attention()

    def forward(self, attn_mask):
        W_Q = self.W_Q(self.Q).view(self.batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        W_K = self.W_K(self.K).view(self.batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        W_V = self.W_V(self.V).view(self.batch_size, -1, self.n_head, self.d_head).transpose(1,2)

        self_attention, soft_mat = self.self_attn(W_Q, W_K, W_V, attn_mask)
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

        self.multi_attn = multi_head_attention(Q, K, V, self.n_head, self.batch_size)

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

    def forward(self, inputs, attn_mask):
        Q = self.m_Q(inputs)
        K = self.m_K(inputs)
        V = self.m_V(inputs)

        before_result = self.first_hidden_state()
        attn_result, soft_mat = self.multi_attn(attn_mask)

        a_n_result = self.add_and_norm(before_result, attn_result)
        
        feed_result = self.feed_forward_function(a_n_result)

        return feed_result, soft_mat

class Encoder(nn.Module):
    def __init__(self, vocab_size, n_seg_type, n_head, batch_size, d_model, seq_len_size):
        super(Encoder, self).__init__()
        self.n_head = n_head
        self.batch_size = batch_size
        self.d_model = d_model
        self.seq_len_size = seq_len_size
        self.vocab_size = vocab_size
        self.n_seg_type = n_seg_type

        self.enc_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len_size + 1, d_model)
        self.seg_emb = nn.Embedding(n_seg_type, d_model)

        self.encoder_layer = Encoder_Layer(inputs, n_head, batch_size, d_model, seq_len_size)
        
    def forward(self, inputs, segments):
        soft_mats = []

        # 이미 padding 된 input이 들어온다
        positions = torch.arange(inputs.size(1), device = inputs.device, dtype = inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq('<pad>')
        positions.masked_fill_(pos_mask, 0)

        new_inputs = self.enc_emb(inputs) + self.pos_emb(positions) + self.seg_emb(segments)
        attn_mask = get_attn_pad_mask(new_inputs, new_inputs, '<pad>')

        result, soft_mat = self.encoder_layer(new_inputs, attn_mask)

        for _ in range(self.n_head-1):
            result, soft_mat = self.encoder_layer(result)
            soft_mats.append(soft_mat)
        return result, soft_mats

class BERT(nn.Module):
    def __init__(self, vocab_size, n_seg_type, n_head, batch_size, d_model, seq_len_size):
        super(BERT, self).__init__()

        # encoder
        self.encoder = Encoder(vocab_size, n_seg_type, n_head, batch_size, d_model, seq_len_size)
        self.linear = nn.Linear(d_model, d_model)
        # activation function
        self.activation = torch.tanh
    
    def forward(self, inputs, segments):
        # encoder 실행
        outputs, attn_probs = self.encoder(inputs)

        # output에서 맨 첫 번째 값(cls token)을 가져온다
        outputs_cls = output[:, 0].contiguous()
        outputs_cls = self.linear(outputs)
        outputs_cls = self.activation(outputs_cls)

        return outputs, outputs_cls, attn_probs

    def save(self, epoch, loss, path):
        torch.save({
            "epoch" : epoch,
            "loss" : loss,
            "state_dict" : self.state_dict()
        }, path)

    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])

        return save["epoch"], save["loss"]

class BERTPretrain(nn.Module):
    def __init__(self, vocab_size, n_seg_type, n_head, batch_size, d_model, seq_len_size):
        super(BERTPretrain, self).__init__()

        self.bert = BERT(vocab_size, n_seg_type, n_head, batch_size, d_model, seq_len_size)
        # for NSP
        self.projection_cls = nn.Linear(d_model, 2, bias = False)
        # for MLM
        self.projection_mask = nn.Linear(d_model, vocab_size, bias = False)
        self.projection_mask.weight = self.bert.encoder.enc_emb.weight

    def forward(self):
        pass