import os
import argparse
import random
import torch
import torch.optim as optim
import torch.nn as nn

# custom imports
from models import CryptoRegressor, Encoder, Attention
from trainer import train, evaluate
from preprocessor import makeSplits

def _makeParser():
    CELL_choices = ["RNN", "LSTM", "GRU"]
    LOSS_choices = ["MSE", "L1"]
    # SCALER_choices = [None, "MinMaxScaler"]
    FEATURE_choices = ['open', 'high', 'low', 'volume', 'ma', 'ema', 'rsi', 'premium', 'funding', 'hist']

    # REQUIRED ARGS
    parser = argparse.ArgumentParser(description='Crypto Regressor.')
    parser.add_argument('datafile', metavar='datafile', type=str, help="data file (expected csv).")
    
    # MODEL
    # TODO: parser.add_argument('--name', type=str, default="CryptoRegressor", help="model name (overrites all other model params).")
    parser.add_argument('--cell', type=str, default='LSTM', choices=CELL_choices, help=f"recurrent cell type, one of {CELL_choices}")
    parser.add_argument('--hidden', type=int, metavar='INT', default=256, help="number of hidden units for RNN encoder")
    parser.add_argument('--nlayers', type=int, metavar='INT', default=2, help="number of layers of the RNN encoder")
    parser.add_argument('--future', type=int, metavar='INT', default=1, help="number of outputs (i.e. how many steps in the future to predict)")
    parser.add_argument('--bi', action='store_true', help="use bidirectional encoder")
    parser.add_argument('--attn', action='store_true', help="use attention")

    # DATA
    # parser.add_argument('--open', action='store_true', help="use open price")
    # parser.add_argument('--close', action='store_true', help="use close price")
    # parser.add_argument('--high', action='store_true', help="use high price")
    # parser.add_argument('--low', action='store_true', help="use low price")
    # parser.add_argument('--volume', action='store_true', help="use volume")
    # parser.add_argument('--ma', action='store_true', help="use moving average of closing price")
    # parser.add_argument('--ema', action='store_true', help="use exponential moving average")
    # parser.add_argument('--rsi', action='store_true', help="use rsi")
    # parser.add_argument('--premium', action='store_true', help="use premium index")
    # parser.add_argument('--funding', action='store_true', help="use funding rate")
    # parser.add_argument('--hist', action='store_true', help="use histogram")

    # parser.add_argument('--scaler', type=str, default=None, choices=SCALER_choices, help=f"data scaler, must be one of {SCALER_choices}")
    parser.add_argument('--features', type=str, metavar='STR', action='append', nargs='+', default=['close'], choices=FEATURE_choices, help=f"features to use (default ['close']), can be one (or many) of {FEATURE_choices}")
    parser.add_argument('--seqlen', type=int, metavar='INT', default=100, help="sequence length")    
    parser.add_argument('--bs', type=int, metavar='INT', default=32, help="batch size")
    parser.add_argument('--split', type=float, metavar='FLOAT', default=.95, help="training split (default 0.95 train).")
    parser.add_argument('--shuffle', action='store_true', help="shuffle data")

    # TRAINING
    parser.add_argument('--loss', type=str, metavar='STR', default='MSE', choices=LOSS_choices, help=f"loss function, must be one of {LOSS_choices}")
    parser.add_argument('--lr', type=float, metavar='FLOAT', default=1e-3, help="initial learning rate")
    parser.add_argument('--epochs', type=int, metavar='FLOAT', default=10, help="max epochs")
    parser.add_argument('--drop', type=float, metavar='FLOAT', default=0, help="dropout rate")
    
    # UTILS
    parser.add_argument('--seed', type=int, metavar='INT', default=None, help="seed")
    parser.add_argument('--cache', type=str, metavar='STR', default="./results", help="cache directory to save models")
    parser.add_argument('--save', action='store_true', help="save training history")
    return parser

def _seedEverything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # ===== Arg Parser =====
    parser = _makeParser()
    args = parser.parse_args()

    # flatten args.features
    features = []
    for f in args.features:
        features += f if isinstance(f, list) else [f]
    args.features = features
    print("Selected features: ", ' | '.join(features))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.seed:
        _seedEverything(args.seed)

    # ===== Load data =====
    iterators, scaler = makeSplits(args, device)
    train_iter, val_iter, test_iter = iterators

    # ===== Create model =====    
    # encoder
    encoder = Encoder(len(args.features), args.hidden, args.nlayers, args.drop, args.bi, args.cell)    
    
    # attention
    attn_dim = args.hidden * (1 + int(args.bi)) # double if bidirectional
    if args.attn:
        attention = Attention(attn_dim, attn_dim, attn_dim)
    else: 
        attention = None

    # classifier
    model = CryptoRegressor(encoder, attention, attn_dim, args.future)
    model = model.to(device)

    # ===== Optimizer & Criterion =====
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = getattr(torch.nn, args.loss+"Loss")(reduction='mean')
    criterion = criterion.to(device)

    # ===== Training =====
    print("Start training ...")
    model.train()
    train(args.epochs, model, optimizer, criterion, train_iter, val_iter, cache=args.cache, save_history=args.save)
    
    # save args parameters
    state = torch.load(os.path.join(args.cache, 'checkpoint.pth.tar'))
    state['args'] = vars(args)
    torch.save(state, os.path.join(args.cache, 'checkpoint.pth.tar'))

    # ===== Testing =====
    model.eval()
    loss, acc = evaluate(model, test_iter, criterion)
    print(f"\t  * Test: Loss {loss.mean():.3f}\tAcc: {acc.mean()*100:.3f}%")

if __name__=='__main__':
    main()