## Introduction

This is the official implementation of our paper [*AP2O: Correcting LLM-Generated Code Errors Type by Type Like Humans via Adaptive Progressive Preference Optimization*]().

## Requirements
- deepspeed 0.17.2
- python 3.11.11 
- torch 2.7.0
- trl 0.14.0
- transformers 4.51.3
- vllm 0.9.2

## Usage

To initiate the preference data self-generation and preference optimization processes, use the following command:

```bash
sh pipe-qwen2.5-coder.sh
```