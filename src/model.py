import nemo.collections.asr as nemo_asr
from .config import config
from omegaconf import OmegaConf

model_name = config["model"]["name"]
model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
model.change_attention_model("rel_pos_local_attn", [128, 128])
model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select
# Set the decoding strategy
decoding_cfg = OmegaConf.create({
    "strategy": config['model']['decoding_strategy'],
})
model.change_decoding_strategy(decoding_cfg)


