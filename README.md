# Exploring the new optimiser: Adan
![image](https://user-images.githubusercontent.com/18441985/187480020-cf8ef768-348a-4eea-bfd3-74e5286a937b.png)

This repo trains an image classifier with [ADAN (Adaptive Nesterov momentum algorithm)](https://arxiv.org/abs/2208.06677) and compares the results with standard Adam.

- We use @lucidrains PyTorch implementation inside a fastai training loop.
- You can also find a fastai's style Adan implementation by Benjamin Warner in fastai_adan.py

## Results

Go to the [W&B report and check the training results](https://wandb.ai/capecape/adan_optimizer/reports/Adan-The-new-optimizer-that-challenges-Adam--VmlldzoyNTQ5NjQ5)
