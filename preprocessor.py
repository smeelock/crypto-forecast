import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler # scalers
import collections

def rawData(args):
    if isinstance(args, dict):
        args = collections.namedtuple("args", args.keys())(*args.values())
    data = pd.read_csv(args.datafile, sep=',')
    # rename columns to match arguments names
    data = data.rename(columns={'MA':'ma', 'EMA 200':'ema', 'RSI':'rsi', 'Premium Index':'premium', 'Funding Rate':'funding', 'Histogram':'hist'})
    return data[args.features+['time']]
    
def makeSplits(args, device):
    """ Make data splits
    Returns: 
        - iterators: tuple (train, val, test)
        - scaler : price scaler (open, close, high, low)
    """
    if isinstance(args, dict):
        args = collections.namedtuple("args", args.keys())(*args.values())
    data = pd.read_csv(args.datafile, sep=',')
    data['time'] = pd.to_datetime(data['time'])
    # rename columns to match arguments names
    data = data.rename(columns={'MA':'ma', 'EMA 200':'ema', 'RSI':'rsi', 'Premium Index':'premium', 'Funding Rate':'funding', 'Histogram':'hist'})

    # scale prices
    scaler = MinMaxScaler(feature_range=(-1, 1))
    col_to_scale = data.columns.isin(['open', 'close', 'high', 'low'])
    data.loc[:, col_to_scale] = scaler.fit_transform(data.loc[:, col_to_scale])
    data = data[args.features]

    N = data.shape[0]
    train = data.iloc[:int(N*args.split)].values
    test = data.iloc[int(N*args.split):].values

    train = torch.from_numpy(train).type(torch.Tensor).to(device)
    test = torch.from_numpy(test).type(torch.Tensor).to(device)

    def __makeBatches(d): # takes data as input
        batches = []
        idx = np.arange(d.shape[0]-args.seqlen-args.future)
        if args.shuffle:
            np.random.shuffle(idx)
        k = 0
        while k < len(idx):
            # init features & labels with the first sequence starting at index i=idx[k] 
            #       (same as in the following 'while' loop)
            features = d[idx[k]:idx[k]+args.seqlen].unsqueeze(0)
            labels = d[idx[k]+args.seqlen:idx[k]+args.seqlen+args.future, 0].unsqueeze(0) # colum 0 is 'close' price
            while len(features) < args.bs:
                k += 1
                if k >= len(idx):
                    break

                i = idx[k]
                feat_start = i
                feat_end = feat_start + args.seqlen
                features = torch.cat((features, d[feat_start : feat_end].unsqueeze(0)), dim=0)

                lbl_start = feat_end
                lbl_end = lbl_start + args.future
                labels = torch.cat((labels, d[lbl_start : lbl_end, 0].unsqueeze(0)), dim=0)
                assert len(features)==len(labels), "Something's wrong: #features != #labels !"
            
            # features=[seqlen, batch, nfeatures]       labels=[batch, future]
            batches.append((features.transpose(0,1), labels)) 
        return batches

    # ... at the moment, validation set = test set
    iterators = (__makeBatches(train), __makeBatches(test), __makeBatches(test))
    return iterators, scaler