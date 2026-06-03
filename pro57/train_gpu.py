"""ACT model fine-tuning on GPU — collects CVAE latent stats."""
import sys, json
from pathlib import Path

import torch, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from act.configuration_act import ACTConfig
from act.defaults import build_act_config, act_config_to_dict
from act.modeling_act import ACTModel
from act.ACTDataset import ACTDataset

# ── Config ──────────────────────────────────────────────
DATA_DIR        = "output/dataset"
PRETRAINED      = "output/train/model.pt"
OUTPUT_DIR      = Path("checkpoints")
EPOCHS          = 10
BATCH_SIZE      = 8
LR              = 1e-5
ACTION_CHUNK    = 8
HIDDEN_DIM      = 512

OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load data ───────────────────────────────────────────
def load_dataset(data_dir):
    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob("data/chunk-*/file-*.parquet"))
    print(f"Parquet files: {len(parquet_files)}")
    import fastparquet, fastparquet.schema as fps
    fps.SchemaHelper.is_required = lambda s, p: False
    images, states, actions = [], [], []
    for pf_path in parquet_files:
        df = fastparquet.ParquetFile(str(pf_path)).to_pandas()
        for _, row in df.iterrows():
            img_path = data_path / row["observation.image"]
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img = img.resize((224, 224))
                img_t = torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0
            else:
                img_t = torch.zeros(3, 224, 224)
            images.append(img_t)
            states.append(torch.tensor(row["observation.state"], dtype=torch.float32))
            actions.append(torch.tensor(row["action"], dtype=torch.float32))
    return {
        "observation.image": torch.stack(images),
        "observation.state": torch.stack(states),
        "action": torch.stack(actions),
    }

def load_norm_stats(data_dir):
    sp = Path(data_dir)/"meta"/"stats.json"
    if not sp.exists():
        return None, None, None, None
    s = json.loads(sp.read_text())
    return (torch.tensor(s["observation.state"]["q01"], dtype=torch.float32),
            torch.tensor(s["observation.state"]["q99"], dtype=torch.float32),
            torch.tensor(s["action"]["q01"], dtype=torch.float32),
            torch.tensor(s["action"]["q99"], dtype=torch.float32))

print("Loading data...")
data = load_dataset(DATA_DIR)
sq01, sq99, aq01, aq99 = load_norm_stats(DATA_DIR)
state_dim = data["observation.state"].shape[1]
action_dim = data["action"].shape[1]
print(f"Samples: {len(data['observation.image'])}, state_dim={state_dim}, action_dim={action_dim}")

# ── Model ───────────────────────────────────────────────
config = build_act_config(state_dim=state_dim, action_dim=action_dim,
                           action_chunk_size=ACTION_CHUNK, hidden_dim=HIDDEN_DIM)
model = ACTModel(config)

if Path(PRETRAINED).exists():
    print(f"Loading pretrained: {PRETRAINED}")
    ckpt = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing: print(f"  Missing keys: {len(missing)}")
    if unexpected: print(f"  Unexpected keys: {len(unexpected)}")

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

dataset = ACTDataset(data, action_chunk_size=ACTION_CHUNK, normalize_images=True,
                     state_q01=sq01, state_q99=sq99, action_q01=aq01, action_q99=aq99)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f"Batches: {len(loader)}")

optimizer = optim.Adam(model.parameters(), lr=LR)
model = model.to(DEVICE)
model.train()

# ── Train ────────────────────────────────────────────────
all_mu, all_log_sigma = [], []
latent_collection_epochs = min(5, max(1, EPOCHS//2))
print(f"\nTraining {EPOCHS} epochs, collecting latent in last {latent_collection_epochs}...")

for epoch in range(EPOCHS):
    total_loss = total_l1 = total_kl = 0.0
    for batch in loader:
        images = batch["observation"]["image"].to(DEVICE)
        states = batch["observation"]["state"].to(DEVICE)
        actions = batch["action"].to(DEVICE)

        optimizer.zero_grad()
        output = model(images, states, action_target=actions, infer_cvae=False)
        predicted = output["action"]
        kl_loss = output.get("kl_loss")
        mu = output.get("mu")
        log_sigma_x2 = output.get("log_sigma_x2")

        if config.use_cvae and mu is not None and log_sigma_x2 is not None:
            if epoch >= EPOCHS - latent_collection_epochs:
                all_mu.append(mu.detach().cpu())
                all_log_sigma.append(log_sigma_x2.detach().cpu())

        l1_loss = F.l1_loss(predicted, actions)
        loss = l1_loss + (kl_loss * config.kl_weight) if kl_loss is not None else l1_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_l1 += l1_loss.item()
        if kl_loss is not None: total_kl += kl_loss.item()

    n = len(loader)
    print(f"Epoch {epoch+1:3d}/{EPOCHS}  loss={total_loss/n:.6f}  L1={total_l1/n:.6f}" +
          (f"  KL={total_kl/n:.6f}" if total_kl>0 else ""))

print("Training done.")

# ── Save ─────────────────────────────────────────────────
final_path = OUTPUT_DIR / "final_model.pt"
if config.use_cvae and len(all_mu) > 0:
    mu_t = torch.cat(all_mu, dim=0)
    ls_t = torch.cat(all_log_sigma, dim=0)
    latent_mu = mu_t.mean(dim=0)
    latent_log_sigma = ls_t.mean(dim=0)
    print(f"Latent: mu={latent_mu.mean():.4f} log_sigma={latent_log_sigma.mean():.4f}")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "inference_latent_mu": latent_mu,
        "inference_latent_log_sigma": latent_log_sigma,
        "config": act_config_to_dict(config),
    }
else:
    checkpoint = model.state_dict()

torch.save(checkpoint, final_path)
print(f"Saved: {final_path} ({final_path.stat().st_size/1024/1024:.1f}MB)")
