# Crytpocurrency Price Forecasting

## Description
As part of Tsinghua Deep Learning Course taught by Prof. Xiaolin Hu (胡晓林) and Prof. Jun Zhu (朱军), we develop a tool to forecast Bitcoin price using Deep Learning.

Using data we collected on purpose for this project from [Tradingview](https://www.tradingview.com/) with different sample frequencies, our best solution consist of a  LSTM recurrent network with Dropout layers achieving an unscaled RMSE (Root Mean Squared Error) of $491 on the 15-minutes dataset.
Over the course of the project we try out multiple architectures with various advanced deep learning techniques such as attention, bidirection, GRU cells and more.
Tests are conducted using the present code and the Optuna library, and they give interesting insights to justify the final choice of model for predicting Bitcoin prices.

More details can be found within the [report](/report/Final_report.pdf "Report") or in the presentation [slides](/report/Presentation_slides.pdf "Slides") which are available online.

## Usage
### main.py
``` 
usage: main.py [-h] [--cell {RNN,LSTM,GRU}] [--hidden INT] [--nlayers INT]
               [--future INT] [--bi] [--attn] [--features STR [STR ...]]
               [--seqlen INT] [--bs INT] [--split FLOAT] [--shuffle]
               [--loss STR] [--lr FLOAT] [--epochs FLOAT] [--drop FLOAT]
               [--seed INT] [--cache STR] [--save]
               datafile

Crypto Regressor.

positional arguments:
  datafile              data file (expected csv).

optional arguments:
  -h, --help            show this help message and exit
  --cell {RNN,LSTM,GRU}
                        recurrent cell type, one of ['RNN', 'LSTM', 'GRU']
  --hidden INT          number of hidden units for RNN encoder
  --nlayers INT         number of layers of the RNN encoder
  --future INT          number of outputs (i.e. how many steps in the future
                        to predict)
  --bi                  use bidirectional encoder
  --attn                use attention
  --features STR [STR ...]
                        features to use (default ['close']), can be one (or
                        many) of ['open', 'high', 'low', 'volume', 'ma',
                        'ema', 'rsi', 'premium', 'funding', 'hist']
  --seqlen INT          sequence length
  --bs INT              batch size
  --split FLOAT         training split (default 0.95 train).
  --shuffle             shuffle data
  --loss STR            loss function, must be one of ['MSE', 'L1']
  --lr FLOAT            initial learning rate
  --epochs FLOAT        max epochs
  --drop FLOAT          dropout rate
  --seed INT            seed
  --cache STR           cache directory to save models
  --save                save training history
```


### optuner.py 
``` 
usage: optuner.py [-h] [--cell {RNN,LSTM,GRU}] [--hidden INT] [--nlayers INT]
                  [--future INT] [--bi] [--attn] [--features STR [STR ...]]
                  [--seqlen INT] [--bs INT] [--split FLOAT] [--shuffle]
                  [--loss STR] [--lr FLOAT] [--epochs INT] [--drop FLOAT]
                  [--seed INT] [--cache STR] [--save]
                  trials datafile

Optuner CryptoRegressor Study. Leave parameters undefined to let Optuna
optimize them.

positional arguments:
  trials                number of trials to run
  datafile              data file (expected csv).

optional arguments:
  -h, --help            show this help message and exit
  --cell {RNN,LSTM,GRU}
                        recurrent cell type, one of ['RNN', 'LSTM', 'GRU']
  --hidden INT          number of hidden units for RNN encoder
  --nlayers INT         number of layers of the RNN encoder
  --future INT          number of outputs (i.e. how many steps in the future
                        to predict)
  --bi                  use bidirectional encoder
  --attn                use attention
  --features STR [STR ...]
                        features to use (default ['close']), can be one (or
                        many) of ['open', 'high', 'low', 'volume', 'ma',
                        'ema', 'rsi', 'premium', 'funding', 'hist']
  --seqlen INT          sequence length
  --bs INT              batch size
  --split FLOAT         training split (default 0.95 train).
  --shuffle             shuffle data
  --loss STR            loss function, must be one of ['MSE', 'L1']
  --lr FLOAT            initial learning rate
  --epochs INT          max epochs
  --drop FLOAT          dropout rate
  --seed INT            seed
  --cache STR           cache directory to save models
  --save                save training history

```
