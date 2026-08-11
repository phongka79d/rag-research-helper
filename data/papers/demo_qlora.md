# Illustrative QLoRA Research Note

## Abstract

This short paper-style note explains QLoRA as an illustrative extension of LoRA for
memory-constrained fine-tuning. It combines low-rank adapters with quantized frozen
base weights.

## Method

QLoRA stores the frozen base model using 4-bit NormalFloat (NF4) quantization and
keeps LoRA adapters in higher precision for training. Double quantization compresses
the quantization constants. Paged optimizers manage memory spikes by moving optimizer
state when necessary. The illustrative adapter setup uses rank r = 16 and scaling
alpha = 32.

## Results

QLoRA can adapt a quantized base model without updating the quantized base weights.
Relative to illustrative LoRA, QLoRA adds quantization and memory-management choices
while preserving the low-rank adapter idea. The resulting adapters can be kept
separately from the frozen quantized base model.
