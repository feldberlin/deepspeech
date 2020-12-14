![Build](https://github.com/feldberlin/deepspeech/workflows/CI/badge.svg)

# DeepSpeech

A minimalistic deep speech implementation.

## Install

Set up a virtualenv, including jupyter:

```bash
bin/install-dev
```

## Experiments

As it turns out, obtaining the original Switchboard and Fisher datasets costs
6000 USD 😱. So instead of attempting to reproduce the original paper, let's
attempt to match the mozilla implementation.

## Common voice

On DeepSpeech 0.9.3 [^1]:

> The acoustic models were trained on American English with synthetic noise
> augmentation and the .pbmm model achieves an 7.06% word error rate on the
> LibriSpeech clean test corpus.

## References

[^1]: [DeepSpeech 0.9.3](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3)
