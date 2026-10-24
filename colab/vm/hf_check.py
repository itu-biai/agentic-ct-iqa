"""Check that the Hugging Face token on the VM can load the gated google/medsiglip-448 (processor and config only)."""
from transformers import AutoConfig, AutoProcessor

try:
    AutoProcessor.from_pretrained("google/medsiglip-448")
    config = AutoConfig.from_pretrained("google/medsiglip-448")
    print("MedSigLIP access OK:", config.model_type)
except Exception as error:
    print("MedSigLIP access FAILED:", type(error).__name__, str(error)[:400])
