"""Export ACT model to ONNX with deterministic CVAE (eps=0)."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from act.configuration_act import ACTConfig
from act.modeling_act import ACTModel

CKPT = "checkpoints/final_model.pt"
OUTPUT = "checkpoints/act_model.onnx"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
config = ACTConfig(**ckpt["config"])
model = ACTModel(config)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Deterministic CVAE: latent = zeros, no sampling noise
class DeterministicACT(torch.nn.Module):
    def __init__(self, base_model, latent_dim):
        super().__init__()
        self.base = base_model
        self.latent_dim = latent_dim

    def forward(self, images, state):
        batch_size = images.shape[0]
        latent = torch.zeros(batch_size, self.latent_dim, device=images.device)

        vision_features = self.base.vision_encoder(images)
        state_features = self.base.state_encoder(state)
        latent_features = self.base.latent_proj(latent).unsqueeze(1)

        encoder_in = torch.cat([latent_features, state_features, vision_features], dim=1)
        seq_len = encoder_in.shape[1]
        pos_embed = self.base.encoder_pos_embed.weight[:seq_len].unsqueeze(0)
        encoder_out = self.base.encoder(encoder_in, pos_embed=pos_embed)

        decoder_pos_embed = self.base.decoder_pos_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoder_in = torch.zeros(batch_size, self.base.config.action_chunk_size,
                                 self.base.config.hidden_dim, device=images.device) + decoder_pos_embed
        decoder_out = self.base.decoder(decoder_in, encoder_out,
                                         decoder_pos_embed=decoder_pos_embed,
                                         encoder_pos_embed=pos_embed)
        action = self.base.action_head(decoder_out)
        return action

det_model = DeterministicACT(model, config.latent_dim)

# Dummy inputs
dummy_img = torch.randn(1, 3, 224, 224)
dummy_state = torch.randn(1, 2)

# Export
torch.onnx.export(
    det_model,
    (dummy_img, dummy_state),
    OUTPUT,
    input_names=["images", "state"],
    output_names=["action"],
    opset_version=14,
    dynamic_axes={
        "images": {0: "batch"},
        "state": {0: "batch"},
        "action": {0: "batch"},
    },
)

print(f"Exported: {OUTPUT} ({Path(OUTPUT).stat().st_size/1024/1024:.1f}MB)")

# Verify
import onnx
onnx_model = onnx.load(OUTPUT)
onnx.checker.check_model(onnx_model)
print("ONNX check: OK")

# Test inference
import onnxruntime as ort
sess = ort.InferenceSession(OUTPUT)
out = sess.run(None, {"images": dummy_img.numpy(), "state": dummy_state.numpy()})
print(f"ONNX output shape: {out[0].shape}")  # should be [1, 8, 3]
print("ONNX inference: OK")
