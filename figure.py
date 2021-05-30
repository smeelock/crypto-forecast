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

def showStudyResults(path, show_max=20):
    """ Show study results as a table """
    filename = os.path.join(path, 'study.pkl')
    study = joblib.load(filename)
    best = study.trials_dataframe().sort_values('value', ascending=True).iloc[:show_max]
    best = best.rename(columns={'params_hidden': 'hidden', 'params_nlayers': 'nlayers'})
    best['rank'] = range(1, best.shape[0]+1)
    best = best.set_index('rank')
    best['duration(s)'] = best['duration'].dt.total_seconds()
    # best[['value', 'duration(s)', 'batch_size', 'bidirectional', 'cell', 'dropout', 'hidden', 'lr', 'nlayers', 'use_attention']]
    return best

def makeBoxplots(studies, names=None):
    """ Plot boxplot for each study """
    boxplots = []
    for i, path in enumerate(studies):
        study = joblib.load(path)
        df = study.trials_dataframe()
        df = df.sort_values('value', ascending=True)
        df['study'] = names[i] if names else i
        df['rank'] = range(1, df.shape[0]+1)
        boxplots.append(df)

    boxplots = pd.concat(boxplots)
    n = boxplots.shape[0] // len(studies)
    best10 = boxplots[boxplots['rank'] < n//10]
    best20 = boxplots[boxplots['rank'] < n//20]
    ax = sns.boxplot(x="study", y="value", data=boxplots)
    ax.set_ylabel("test loss")
    ax.figure.savefig("boxplot_all.png")

    plt.figure()
    ax10 = sns.boxplot(x="study", y="value", data=best10)
    ax10.set_ylabel("test loss")
    ax10.figure.savefig("boxplot_best10.png")

    plt.figure()
    ax20 = sns.boxplot(x="study", y="value", data=best20)
    ax20.set_ylabel("test loss")
    ax20.figure.savefig("boxplot_best20.png")


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