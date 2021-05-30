import os
import collections
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # scalers
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme('notebook')

from preprocessor import makeSplits, rawData

def plotLoss(filepath):
    state = torch.load(os.path.join(filepath, 'checkpoint.pth.tar'))
    assert 'history' in state.keys(), "Incomplete state file, must contain 'history'."

    history = state['history']
    train_hist, val_hist = history

    df = pd.DataFrame({
        'loss': train_hist['loss'], 
        'acc': train_hist['acc'],
        'step': range(len(train_hist['loss'])),
        'type': "train",
    })
    df = pd.concat([df, pd.DataFrame({
        'loss': val_hist['loss'], 
        'acc': val_hist['acc'],
        'step': range(len(val_hist['loss'])),
        'type': "validation",
    })])

    plt.figure()
    return sns.lineplot(data=df, x='step', y='loss', hue='type')

def plotTrialsLoss(trials_path):
    trial_no = 0
    df = pd.DataFrame(columns=['loss', 'acc', 'step', 'trial'])
    while os.path.isdir(f"{trials_path}/trial{trial_no}"):
        state = torch.load(os.path.join(f"{trials_path}/trial{trial_no}", "checkpoint.pth.tar"), map_location='cpu')
        _, val_hist = state['history']

        new_df = pd.DataFrame({
            'loss': val_hist['loss'],
            'acc': val_hist['acc'],
            'step': range(len(val_hist['loss'])),
            'trial': trial_no,
        })
        df = pd.concat([df, new_df])

        trial_no += 1

    df = df[df['step'] > 1] # skip first step because is weird

    plt.figure('All Trials', figsize=(20,13))
    ax = sns.lineplot(data=df, x='step', y='loss', hue='trial')
    ax.set(yscale='log')
    fig.savefig("all_trials.png")
    return ax
    plt.figure()
    return sns.lineplot(data=df, x='step', y='loss', hue='trial')


def plotComparison(filepath, datafile):
    # comparison real/predicted
    state = torch.load(os.path.join(filepath, 'checkpoint.pth.tar'))
    assert 'model' in state.keys(), "Incomplete state file, must contain 'model'."
    assert 'args' in state.keys(), "Incomplete state file, must contain 'args'."

    model = state['model']
    args = state['args']
    args['split'] = 0 # only testing, no training data
    args['shuffle'] = False # don't shuffle
    args['datafile'] = datafile
    args = collections.namedtuple("args", args.keys())(*args.values())
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    iterators, _ = makeSplits(args, device)
    _, val_iter, _ = iterators

    df = rawData(args)[['time', 'close']]
    df = df.rename(columns={'close': 'price'})
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df['price'] = scaler.fit_transform(df['price'].values.reshape(-1, 1))
    df['step'] = range(df.shape[0])
    df['type'] = 'real'

    timestep = args.future
    model.eval()
    with torch.no_grad():
        for batch in val_iter:
            features, labels = batch
            logits = model.forward(features)
            logits = logits.to('cpu')
            # logits = scaler.inverse_transform(logits)
            
            batch_bs, batch_future = logits.shape
            t = np.array([range(timestep+i, timestep+i+batch_future) for i in range(batch_bs)]).flatten()
            df = pd.concat([df, pd.DataFrame({
                'step': t,
                # 'time': df['time'].iloc[t],
                'price': logits.flatten(),
                'type': 'predicted',
            })])
            timestep += args.bs

    plt.figure()
    return sns.lineplot(data=df, x='step', y='price', hue='type')