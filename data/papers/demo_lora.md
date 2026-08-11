# Illustrative LoRA Research Note

## Abstract

This short paper-style note explains Low-Rank Adaptation (LoRA) as an illustrative
parameter-efficient fine-tuning method. The pretrained model remains fixed while a
small low-rank update is trained for a target task.

## Method

Let the pretrained weight matrix be W0. LoRA keeps W0 frozen and learns a weight
update Delta W = B A, where B and A are low-rank matrices. This illustrative setup
uses rank r = 8 and scaling alpha = 16. During training, only B and A receive
gradients. The example optimizer uses learning rate 3e-4 and batch size 32.

## Results

Compared with full fine-tuning, the illustrative method stores only adapter matrices
for each task. The base model can be reused for multiple tasks by loading different
adapters. This avoids keeping one complete copy of the model per task.
