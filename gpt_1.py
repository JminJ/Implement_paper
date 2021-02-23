import torch
import torch.nn as nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, input, input_size, embedding_size):
        # input은 bpe 알고리즘으로 전처리 및 0으로 padding 되어있을 것입니다.
        self.input = input
        self.input_len = input.size()
        self.input_size = input_size
        self.embedding_size = embedding_size

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
        
        