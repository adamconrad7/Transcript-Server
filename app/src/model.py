import nemo.collections.asr as nemo_asr
from .config import config
from omegaconf import OmegaConf
import logging

logger = logging.getLogger(__name__)

model = None

def initialize_model():
    global model
    if model is None:
        model_name = config["model"]["name"]
        logger.info(f"Initializing model: {model_name}")

        try:
            # Load the model without any config overrides initially
            model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name)
            logger.info("Model loaded successfully")

            # Now apply the decoding strategy
            decoding_cfg = OmegaConf.create({
                "strategy": config['model']['decoding_strategy'],
            })
            model.change_decoding_strategy(decoding_cfg)
            logger.info(f"Decoding strategy applied: {config['model']['decoding_strategy']}")

            # Apply other model-specific configurations
            model.change_attention_model("rel_pos_local_attn", [128, 128])
            model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select

            model.eval()
            
            if config.get('use_cuda', False):
                model = model.cuda()

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    return model

def get_model():
    return initialize_model()
