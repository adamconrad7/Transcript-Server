import nemo.collections.asr as nemo_asr

# Load the Parakeet-CTC model
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")

# Export to ONNX
model.export("parakeet_ctc.onnx")
