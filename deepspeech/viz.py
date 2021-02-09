import numpy as np
import matplotlib.pyplot as plt
import torch

from deepspeech import datasets, predict


def ctc_batch(m, batch, cfg):
    x, xn, y, yn = datasets.batch(cfg)(batch)
    yhat, loss = predict.predict(m, x, xn, y, yn)
    print(loss)
    ctc(m, x, yhat)


def ctc(m, x, yhat):
    batch_size = yhat.shape[0]
    fig, axs = plt.subplots(1, batch_size * 2, figsize=(20, 30))
    for i in range(batch_size):
        preds = torch.exp(yhat[i]).cpu().numpy()
        sound = x[i].log2().cpu().numpy().T
        axs[i*2].matshow(preds, cmap=plt.cm.Blues)
        axs[i*2-1].imshow(sound, aspect='auto')

