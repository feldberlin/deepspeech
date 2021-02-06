from deepspeech import model, train, utils


def load(run_path):
    "Load config and model from wandb"
    p, ptrain = utils.load_wandb_cfg(run_path)
    p, ptrain = model.HParams(**p), train.HParams(**ptrain)
    return utils.load_chkpt(model.DeepSpeech(p), run_path), ptrain
