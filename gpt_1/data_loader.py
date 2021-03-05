import torch
from torchtext import data
import pandas as pd
from eunjeon import Mecab

mecab = Mecab()

class Dataloader_fn(object):
  def __init__(self, train_fn, batch_size = 64, valid_ratio = .2, device = -1, min_freq = 5, max_vocab = 99999, use_eos = False, shuffle = True):
    # train_fn : train dataset path, max_vocab : max vocab size, min_freq : minimum frequency for loaded word
    super(Dataloader_fn, self).__init__()

    self.label = data.Field(
        sequential = False,
        use_vocab = True,
        unk_token = False
    )
    self.text = data.Field(
        use_vocab = True,
        batch_first = True,
        include_lengths = False,
        tokenize = mecab.morphs,
        init_token = '<sos>',
        eos_token = '<eos>'
    )

    train, valid = data.TabularDataset(
        path = '/content/drive/MyDrive/korean_smentic_dataset_ver_one.csv',
        format = 'csv',
        fields = [
                  ('text', self.text),
                  ('label', self.label),
        ],
    ).split(split_ratio = (1-valid_ratio))

    self.train_loader, self.test_loader = data.BucketIterator.splits(
        (train, valid),
        batch_size = batch_size,
        device = 'cuda:%d' % device if device >= 0 else 'cpu',
        shuffle = shuffle,
        sort_key = lambda x: len(x.text),
        sort_within_batch = True, 
    )
    
    self.label.build_vocab(train)
    self.text.build_vocab(train, max_size = max_vocab, min_freq = min_freq)