import torch
from torchtext import data
import pandas as pd

class Dataloader_Pre_train(object):
  def __init__(self, train_fn, batch_size = 64, device = -1, min_freq = 5, max_vocab = 99999, use_eos = True, shuffle = True):
    # train_fn : train dataset path, max_vocab : max vocab size, min_freq : minimum frequency for loaded word
    super(Dataloader_Pre_train, self).__init__()

    self.pre_t = data.Field(
        use_vocab = True,
        batch_first = True,
        include_lengths = False,
        init_token = '<BOS>',
        eos_token = '<EOS>',
        pad_token= '<PAD>'
    )

    train = data.TabularDataset(
        path = 'Implement_paper\gpt_1\kowiki.sentence_piece_text.json', # change your data path
        format = 'json', # change your data format
        fields = {
                  'document' : self.pre_t,
        },
    )

    self.train_loader = data.BucketIterator(
        train,
        batch_size = batch_size,
        device = 'cuda:%d' % device if device >= 0 else 'cpu',
        shuffle = shuffle,
        sort_key = lambda x: len(x.text),
        sort_within_batch = True, 
    )
    
    self.text.build_vocab(train, max_size = max_vocab, min_freq = min_freq)       