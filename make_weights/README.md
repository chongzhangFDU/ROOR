# Preparing the pre-trained models

1. Please manually download the pre-trained weights and tokenizers of [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base), [GeoLayoutLM](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/blob/main/DocumentUnderstanding/GeoLayoutLM/README.md#model-checkpoints), [BERT](https://huggingface.co/google-bert/bert-base-uncased) and [BART](https://huggingface.co/facebook/bart-base). 
2. Some experiments require longer input context, which may exceed the maximum input length of these pre-trained models. Therefore, 
   1. For LayoutLMv3, we expand the max input length from 512 to 2k. Please run `v3.py` to make the weights, and manually modify:
      1. the value of `max_position_embeddings` from 514 to 2056 in `/path/to/layoutlmv3-{base,large}-2048/config.json`;
      2. the value of `model_max_length` from 512 to 2048 in `/path/to/layoutlmv3-{base,large}-2048/tokenizer_config.json `. 
   2. For GeoLayoutLM, we expand the max input length from 512 to 1k. Please run `geo.py` to make the weights, and manually modify:
      1. the value of `max_position_embeddings` from 512 to 1024 in `rore/configs/config.json`. 
3. The QA experiments require a generative variant of LayoutLMv3. Please run `gen.py` to make the initial checkpoint, with the use of the weights of `layoutlmv3-{base,large}-2048`. Please add `/path/to/ROOR/rore` to the environment variable `PYTHONPATH` to use the model implementation. 