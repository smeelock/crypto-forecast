import argparse
import os
import optuna
import torch
import torch.optim as optim
import joblib # pkl files
import collections # transform dict in object

# custom imports
from trainer import train, evaluate
from preprocessor import makeSplits
from models import Encoder, Attention, CryptoRegressor

def _makeParser():
    CELL_choices = ["RNN", "LSTM", "GRU"]
    LOSS_choices = ["MSE", "L1"]
    FEATURE_choices = ['open', 'high', 'low', 'volume', 'ma', 'ema', 'rsi', 'premium', 'funding', 'hist']
    
    parser = argparse.ArgumentParser(description='Optuner CryptoRegressor Study. Leave parameters undefined to let Optuna optimize them.')

    # REQUIRED ARGS
    parser.add_argument('trials', type=int, help='number of trials to run')
    parser.add_argument('datafile', metavar='datafile', type=str, help="data file (expected csv).")
    
    # MODEL
    # TODO: parser.add_argument('--name', type=str, default="CryptoRegressor", help="model name (overrites all other model params).")
    parser.add_argument('--cell', type=str, default='LSTM', choices=CELL_choices, help=f"recurrent cell type, one of {CELL_choices}")
    parser.add_argument('--hidden', type=int, metavar='INT', default=None, help="number of hidden units for RNN encoder")
    parser.add_argument('--nlayers', type=int, metavar='INT', default=None, help="number of layers of the RNN encoder")
    parser.add_argument('--future', type=int, metavar='INT', default=1, help="number of outputs (i.e. how many steps in the future to predict)")
    parser.add_argument('--bi', action='store_true', help="use bidirectional encoder")
    parser.add_argument('--attn', action='store_true', help="use attention")

    # DATA
    parser.add_argument('--features', type=str, metavar='STR', action='append', nargs='+', default=['close'], choices=FEATURE_choices, help=f"features to use (default ['close']), can be one (or many) of {FEATURE_choices}")
    parser.add_argument('--seqlen', type=int, metavar='INT', default=None, help="sequence length")    
    parser.add_argument('--bs', type=int, metavar='INT', default=None, help="batch size")
    parser.add_argument('--split', type=float, metavar='FLOAT', default=.95, help="training split (default 0.95 train).")
    parser.add_argument('--shuffle', action='store_true', help="shuffle data")

    # TRAINING
    parser.add_argument('--loss', type=str, metavar='STR', default='MSE', choices=LOSS_choices, help=f"loss function, must be one of {LOSS_choices}")
    parser.add_argument('--lr', type=float, metavar='FLOAT', default=None, help="initial learning rate")
    parser.add_argument('--epochs', type=int, metavar='INT', default=30, help="max epochs")
    parser.add_argument('--drop', type=float, metavar='FLOAT', default=None, help="dropout rate")
    
    # UTILS
    parser.add_argument('--seed', type=int, metavar='INT', default=None, help="seed")
    parser.add_argument('--cache', type=str, metavar='STR', default="./optuner", help="cache directory to save models")
    parser.add_argument('--save', action='store_true', help="save training history")
    return parser

def objective(trial, args):
    config = {
        'trials': args.trials,
        'datafile': args.datafile,

        # model
        'cell': trial.suggest_categorical('cell', ['LSTM', 'GRU', 'RNN']) if args.cell is None else args.cell,
        'hidden': trial.suggest_categorical('hidden', [64, 128, 256, 512, 1024]) if args.hidden is None else args.hidden,
        'nlayers': trial.suggest_categorical('nlayers', [1, 2, 3, 4, 5]) if args.nlayers is None else args.nlayers,
        
        # training
        'bs': trial.suggest_categorical('bs', [16, 32, 64, 128, 256, 512, 1024]) if args.bs is None else args.bs,
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1)  if args.lr is None else args.lr,
        'drop': trial.suggest_uniform('drop', 0, 0.9)  if args.drop is None else args.drop,
        'seqlen': trial.suggest_int('seqlen', 1, 501, 20)  if args.seqlen is None else args.seqlen, # 1 to 500 step of 20

        # fixed params
        'split': args.split,
        'shuffle': args.shuffle,
        'features': args.features,
        'loss': args.loss,
        'epochs': args.epochs,
        'bi': args.bi,
        'attn': args.attn,
        'future': args.future,
        'seed': args.seed,
        'cache': os.path.normpath(f"{args.cache}/trial{trial.number}/"),
        'save': args.save,
    }
    # transform dict to object (to keep the convention args.parameter and not args['parameter'])
    config = collections.namedtuple("config", config.keys())(*config.values())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.seed:
        _seedEverything(config.seed)

    # ===== Load data =====
    iterators, scaler = makeSplits(config, device)
    train_iter, val_iter, test_iter = iterators

    # ===== Create model =====    
    # encoder
    encoder = Encoder(len(config.features), config.hidden, config.nlayers, config.drop, config.bi, config.cell)    
    
    # attention
    attn_dim = config.hidden * (1 + int(config.bi)) # double if bidirectional
    attention = Attention(attn_dim, attn_dim, attn_dim) if config.attn else None

    # classifier
    model = CryptoRegressor(encoder, attention, attn_dim, config.future)
    model = model.to(device)

    # ===== Optimizer & Criterion =====
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = getattr(torch.nn, config.loss+"Loss")(reduction='mean')
    criterion = criterion.to(device)

    # ===== Training =====
    print("Start training...")
    model.train()
    train(config.epochs, model, optimizer, criterion, train_iter, val_iter, cache=config.cache, save_history=config.save, trial=trial)

    # ===== Testing =====
    model.eval()
    loss, acc = evaluate(model, test_iter, criterion)
    print(f"\t  * Test: Loss {loss.mean():.3f}\tAcc: {acc.mean()*100:.3f}%")
    
    return loss.mean() # return val_loss

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

    # study
    study = optuna.create_study(study_name="Optuner CryptoRegressor", direction="minimize") # minimize val loss
    objective_fct = lambda trial: objective(trial, args)
    study.optimize(objective_fct, n_trials=args.trials)

    # print stats
    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # save study
    filename = os.path.normpath(f"{args.cache}/study.pkl")
    joblib.dump(study, filename)

if __name__=='__main__':
    main()
