import torch.cuda.amp as amp

from deepspeech import model, train, utils


def load(run_path):
    "Load config and model from wandb"
    p, ptrain = utils.load_wandb_cfg(run_path)
    p, ptrain = model.HParams(**p), train.HParams(**ptrain)
    return utils.load_chkpt(model.DeepSpeech(p), run_path), ptrain


def predict(m, x, nx):
    device = torch.cuda.current_device()
    with amp.autocast(enabled=m.cfg.mixed_precision):
        m = m.to(device)
        x = x.to(device)
        nx = nx.to(device)
        with torch.set_grad_enabled(False):
            yhat, _ = m.forward(x, nx)
        return yhat
