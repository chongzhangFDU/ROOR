import torch
from model.layoutlm_v3.generative_qa import LayoutLMv3ForConditionalGeneration
from model.layoutlm_v3.configuration_layoutlmv3 import LayoutLMv3Config
from model.layoutlm_v3.tokenization_layoutlmv3 import LayoutLMv3Tokenizer
from model.layoutlm_v3.modeling_layoutlmv3 import LayoutLMv3Model
from transformers import BartConfig, BartModel, BartTokenizer


config = LayoutLMv3Config.from_pretrained('/path/to/layoutlmv3-base-2048')
bart_config = BartConfig.from_pretrained('/path/to/bart-base')
model = LayoutLMv3ForConditionalGeneration(config=config, bart_config=bart_config)
bart_model = BartModel.from_pretrained('/path/to/bart-base')
model.layoutlmv3.decoder.load_state_dict(bart_model.decoder.state_dict())
layoutlmv3_model = LayoutLMv3Model.from_pretrained('/path/to/layoutlmv3-base-2048')
model.layoutlmv3.encoder.load_state_dict(layoutlmv3_model.state_dict())
# model.config.decoder_start_token_id = model.config.eos_token_id
# model.config.is_encoder_decoder = True
# model.config.use_cache = True
torch.save(model, 'model.pt')
