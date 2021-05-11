import os
import time
import shutil
import numpy as np
import torch
import torch.nn.functional as F # softmax
import optuna

def train(max_epoch, model, optimizer, criterion, train_iterator, val_iterator, cache="./results", save_history=False, trial=None):
    """ Trainer function for CrytpoRegressor. """
    if not os.path.isdir(cache):
        os.makedirs(cache) # mkdir but recursively
        print(f"Created cache directory {cache}.")

    best_val_acc = float('inf')
    train_history =  {'loss':[], 'acc': []}
    val_history = {'loss': [], 'acc': []}

    for epoch in range(max_epoch):
        model.train() # make sure model is in train mode

        # timer
        start_time = time.time()

        # train one epoch
        train_loss, train_acc = _trainOneEpoch(model, train_iterator, optimizer, criterion)

        # evaluate
        val_loss, val_acc = evaluate(model, val_iterator, criterion)

        # stats
        duration = time.time() - start_time
        train_history['loss'].append(train_loss.mean())
        train_history['acc'].append(train_acc.mean())
        val_history['loss'].append(val_loss.mean())
        val_history['acc'].append(val_acc.mean())
        
        is_best = ((val_acc).mean() > best_val_acc)
        best_val_acc = max(best_val_acc, val_acc.mean()) # update best validation accuracy

        # save checkpoint
        state = {
            'model': model,
            # 'epoch': epoch + 1,
            # 'state_dict': model.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            'best_acc': best_val_acc,
        }
        if save_history:
            state['history'] = (train_history, val_history)
        filename = os.path.normpath(f"{cache}/checkpoint.pth.tar")
        _saveCheckpoint(state, is_best, filename)

        # verbose
        display = [
            f"Epoch: [{epoch+1:03}/{max_epoch}]",
            f"Time: {duration:.3f}s",
            f"Loss: {train_loss.mean():.3e}",
            f"Acc: {train_acc.mean()*100:.2f}%",
            f"Loss(val): {val_loss.mean():.2e}",
            f"Acc(val): {val_acc.mean()*100:.2f}%"
        ]
        print('   '.join(display))

        # report to Optuna trial
        if trial is not None:
            trial.report(val_loss.mean(), epoch)
            if trial.should_prune(): # and early stop if necessary
                raise optuna.exceptions.TrialPruned()


def evaluate(model, iterator, criterion):
    epoch_loss = []
    epoch_acc = []

    model.eval() # deactivate dropout

    with torch.no_grad():
        for batch in iterator:
            features, labels = batch
            logits = model.forward(features)
            loss = criterion(logits.view(-1, model.output_dim), labels)
            acc = _accuracy(logits, labels)

            epoch_loss.append(loss.item())
            epoch_acc.append(acc.item())

    return np.array(epoch_loss), np.array(epoch_acc)

def _trainOneEpoch(model, iterator, optimizer, criterion):
    epoch_loss = []
    epoch_acc = []

    for batch in iterator:
        features, labels = batch
        optimizer.zero_grad()
        
        logits = model.forward(features) # text, text_lengths = batch.text
        loss = criterion(logits.view(-1, model.output_dim), labels)
        acc = _accuracy(logits, labels)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        epoch_acc.append(acc.item())

    return np.array(epoch_loss), np.array(epoch_acc)

def _accuracy(y_pred, y_true):
    acc = (y_true - y_pred) / y_true
    acc = 1 - acc.abs().mean()
    return acc

def _saveCheckpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(
            os.path.dirname(filename), 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

