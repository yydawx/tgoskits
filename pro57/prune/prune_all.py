"""
Single-strategy pruning for ACT model.
Each variant changes exactly ONE dimension.
"""
import torch
import numpy as np
import os, sys, json, copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from act.modeling_act import ACTModel
from act.configuration_act import ACTConfig


def slice_linear_weight(w, bias, new_out, new_in):
    """Slice Linear weight from (old_out, old_in) to (new_out, new_in)."""
    return w[:new_out, :new_in], bias[:new_out] if bias is not None else None


def slice_attn_weights(sd, prefix, old_dim, new_dim):
    """Slice MultiheadAttention weights in-place. Returns modified sd dict."""
    # in_proj_weight: (3*old_dim, old_dim) → (3*new_dim, new_dim)
    # Q[0:old_dim], K[old_dim:2*old_dim], V[2*old_dim:3*old_dim]
    # Take first new_dim rows from each section, and first new_dim cols
    w = sd[f'{prefix}.in_proj_weight']
    q = w[0:new_dim, 0:new_dim]           # first new_dim rows of Q
    k = w[old_dim:old_dim+new_dim, 0:new_dim]  # first new_dim rows of K
    v = w[2*old_dim:2*old_dim+new_dim, 0:new_dim]  # first new_dim rows of V
    sd[f'{prefix}.in_proj_weight'] = torch.cat([q, k, v], dim=0)

    # in_proj_bias: same logic
    b = sd[f'{prefix}.in_proj_bias']
    sd[f'{prefix}.in_proj_bias'] = torch.cat([
        b[0:new_dim],
        b[old_dim:old_dim+new_dim],
        b[2*old_dim:2*old_dim+new_dim]
    ])

    # out_proj: (old_dim, old_dim) → (new_dim, new_dim)
    sd[f'{prefix}.out_proj.weight'] = sd[f'{prefix}.out_proj.weight'][:new_dim, :new_dim]
    sd[f'{prefix}.out_proj.bias'] = sd[f'{prefix}.out_proj.bias'][:new_dim]

    return sd


def prune_model(ckpt, strategy, value):
    """Create pruned model with single-strategy change."""
    cfg_dict = ckpt['config']
    sd_orig = {k: v.clone() for k, v in ckpt['model_state_dict'].items()}
    sd = {k: v.clone() for k, v in ckpt['model_state_dict'].items()}

    old_hidden = cfg_dict['hidden_dim']
    old_ffn = cfg_dict['dim_feedforward']
    old_enc = cfg_dict['num_encoder_layers']
    old_dec = cfg_dict['num_decoder_layers']
    old_heads = cfg_dict['num_attention_heads']

    new_hidden = old_hidden
    new_ffn = old_ffn
    new_enc = old_enc
    new_dec = old_dec
    new_heads = old_heads

    if strategy == 'enc':      new_enc = value
    elif strategy == 'dec':    new_dec = value
    elif strategy == 'hidden': new_hidden = value
    elif strategy == 'ffn':    new_ffn = value
    elif strategy == 'heads':  new_heads = value
    else: raise ValueError(f"Unknown strategy: {strategy}")

    # Slice weights based on strategy
    if strategy == 'hidden' and new_hidden != old_hidden:
        d = new_hidden
        # Vision encoder
        vep = 'vision_encoder.encoder_img_feat_input_proj'
        sd[f'{vep}.weight'], sd[f'{vep}.bias'] = slice_linear_weight(
            sd[f'{vep}.weight'], sd.get(f'{vep}.bias'), d, old_hidden)
        # State encoder
        for ln in ['0', '2']:
            sep = f'state_encoder.encoder.{ln}'
            sd[f'{sep}.weight'], sd[f'{sep}.bias'] = slice_linear_weight(
                sd[f'{sep}.weight'], sd.get(f'{sep}.bias'), d, d)

        # Encoder layers
        for l in range(old_enc):
            pre = f'encoder.layers.{l}'
            # self_attn
            sa = f'{pre}.self_attn'
            sd = slice_attn_weights(sd, sa, old_hidden, d)
            # FFN: Linear(d, old_ffn), Linear(old_ffn, d)
            sd[f'{pre}.linear1.weight'], sd[f'{pre}.linear1.bias'] = slice_linear_weight(
                sd[f'{pre}.linear1.weight'], sd[f'{pre}.linear1.bias'], old_ffn, d)
            sd[f'{pre}.linear2.weight'], sd[f'{pre}.linear2.bias'] = slice_linear_weight(
                sd[f'{pre}.linear2.weight'], sd[f'{pre}.linear2.bias'], d, old_ffn)
            # LayerNorms
            for ln in ['norm1', 'norm2']:
                k = f'{pre}.{ln}.weight'
                if k in sd: sd[k] = sd[k][:d]
                bk = f'{pre}.{ln}.bias'
                if bk in sd: sd[bk] = sd[bk][:d]

        # Decoder layers
        for l in range(old_dec):
            pre = f'decoder.layers.{l}'
            for attn_name in ['self_attn', 'multihead_attn']:
                sd = slice_attn_weights(sd, f'{pre}.{attn_name}', old_hidden, d)
            sd[f'{pre}.linear1.weight'], sd[f'{pre}.linear1.bias'] = slice_linear_weight(
                sd[f'{pre}.linear1.weight'], sd[f'{pre}.linear1.bias'], old_ffn, d)
            sd[f'{pre}.linear2.weight'], sd[f'{pre}.linear2.bias'] = slice_linear_weight(
                sd[f'{pre}.linear2.weight'], sd[f'{pre}.linear2.bias'], d, old_ffn)
            for ln in ['norm1', 'norm2', 'norm3']:
                k = f'{pre}.{ln}.weight'
                if k in sd: sd[k] = sd[k][:d]
                bk = f'{pre}.{ln}.bias'
                if bk in sd: sd[bk] = sd[bk][:d]

        # Decoder norm, action encoder (CVAE), latent projection, action head
        for k in ['decoder.norm.weight', 'decoder.norm.bias',
                   'action_encoder.weight', 'action_encoder.bias',
                   'vae_output_proj.weight', 'vae_output_proj.bias',
                   'latent_query.weight']:
            if k in sd:
                if 'weight' in k and sd[k].dim() == 2:
                    sd[k] = sd[k][:, :d] if sd[k].shape[1] == old_hidden else sd[k][:d, :]
                elif sd[k].dim() == 1:
                    sd[k] = sd[k][:d]
            elif k.replace('weight','bias') in sd:
                pass  # bias handled with weight

        # latent_proj: Linear(latent_dim, hidden_dim) → weight=(hidden, latent)
        # hidden_dim is out_features → slice rows
        if 'latent_proj.weight' in sd:
            sd['latent_proj.weight'] = sd['latent_proj.weight'][:d, :]
        if 'latent_proj.bias' in sd:
            sd['latent_proj.bias'] = sd['latent_proj.bias'][:d]
        # action_head: Linear(hidden, action_dim) → weight=(action_dim, hidden) → slice cols
        if 'action_head.weight' in sd:
            sd['action_head.weight'] = sd['action_head.weight'][:, :d]

        # Position embeddings: (num_pos, hidden) → slice cols
        for pk in ['encoder_pos_embed.weight', 'decoder_pos_embed.weight']:
            if pk in sd:
                sd[pk] = sd[pk][:, :d]

        # Vision encoder pos embed output: slice appropriately
        for pk in list(sd.keys()):
            if 'cam_feat_pos_embed' in pk:
                sd[pk] = sd[pk][:d]

        # Vision encoder proj: Conv2d(in_ch, hidden, 1) → weight=(hidden, in_ch, 1, 1)
        vep = 'vision_encoder.encoder_img_feat_input_proj'
        if f'{vep}.weight' in sd:
            sd[f'{vep}.weight'] = sd[f'{vep}.weight'][:d, :, :, :]
        if f'{vep}.bias' in sd:
            sd[f'{vep}.bias'] = sd[f'{vep}.bias'][:d]

        # State encoder Linear layers: (state_dim→hidden) → slice rows
        for ln in ['0', '2']:
            sep = f'state_encoder.encoder.{ln}'
            if f'{sep}.weight' in sd:
                sd[f'{sep}.weight'] = sd[f'{sep}.weight'][:d, :]
            if f'{sep}.bias' in sd:
                sd[f'{sep}.bias'] = sd[f'{sep}.bias'][:d]

        # CVAE action_encoder: Linear(chunk*action, hidden) → (hidden, chunk*action) → slice rows
        if 'action_encoder.weight' in sd:
            sd['action_encoder.weight'] = sd['action_encoder.weight'][:d, :]
        if 'action_encoder.bias' in sd:
            sd['action_encoder.bias'] = sd['action_encoder.bias'][:d]
        # vae_output_proj: Linear(hidden, latent*2) → (latent*2, hidden) → slice cols!
        if 'vae_output_proj.weight' in sd:
            sd['vae_output_proj.weight'] = sd['vae_output_proj.weight'][:, :d]
        if 'vae_output_proj.bias' in sd:
            pass  # bias is latent*2, unchanged

        # latent_query: Embedding(1, hidden) → weight=(1, hidden) → slice cols
        if 'latent_query.weight' in sd:
            sd['latent_query.weight'] = sd['latent_query.weight'][:, :d]

        # Encoder/Decoder norm (top-level)
        for nk in ['encoder.norm', 'decoder.norm']:
            for sfx in ['weight', 'bias']:
                k = f'{nk}.{sfx}'
                if k in sd: sd[k] = sd[k][:d]

    elif strategy == 'ffn' and new_ffn != old_ffn:
        d = new_ffn
        # Encoder
        for l in range(old_enc):
            pre = f'encoder.layers.{l}'
            sd[f'{pre}.linear1.weight'], sd[f'{pre}.linear1.bias'] = slice_linear_weight(
                sd[f'{pre}.linear1.weight'], sd[f'{pre}.linear1.bias'], d, old_hidden)
            sd[f'{pre}.linear2.weight'], sd[f'{pre}.linear2.bias'] = slice_linear_weight(
                sd[f'{pre}.linear2.weight'], sd[f'{pre}.linear2.bias'], old_hidden, d)
        # Decoder
        for l in range(old_dec):
            pre = f'decoder.layers.{l}'
            sd[f'{pre}.linear1.weight'], sd[f'{pre}.linear1.bias'] = slice_linear_weight(
                sd[f'{pre}.linear1.weight'], sd[f'{pre}.linear1.bias'], d, old_hidden)
            sd[f'{pre}.linear2.weight'], sd[f'{pre}.linear2.bias'] = slice_linear_weight(
                sd[f'{pre}.linear2.weight'], sd[f'{pre}.linear2.bias'], old_hidden, d)

    # Build new model
    new_cfg = ACTConfig(
        state_dim=cfg_dict['state_dim'], action_dim=cfg_dict['action_dim'],
        action_chunk_size=cfg_dict['action_chunk_size'],
        n_action_steps=cfg_dict['action_chunk_size'],
        hidden_dim=new_hidden, num_attention_heads=new_heads,
        num_encoder_layers=new_enc, num_decoder_layers=new_dec,
        dim_feedforward=new_ffn,
        latent_dim=cfg_dict['latent_dim'],
        use_cvae=cfg_dict['use_cvae'], kl_weight=cfg_dict['kl_weight'],
        use_temporal_ensembling=False,
    )
    model = ACTModel(new_cfg)
    model.eval()
    model.set_inference_latent(ckpt['inference_latent_mu'], ckpt['inference_latent_log_sigma'])
    # Manual weight loading: skip mismatched shapes
    model_sd = model.state_dict()
    matched, skipped = [], []
    for k in list(sd.keys()):
        if k in model_sd and sd[k].shape == model_sd[k].shape:
            model_sd[k] = sd[k]
            matched.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(model_sd, strict=True)
    model._sample_latent = lambda mu, ls: mu
    return model, new_cfg, sd, matched, skipped


# Single-strategy variants to test
SINGLE_VARIANTS = [
    # Encoder layer pruning
    ('enc3',  'enc',    3),
    ('enc2',  'enc',    2),
    ('enc1',  'enc',    1),
    # Hidden dim reduction
    ('h384',  'hidden', 384),
    ('h256',  'hidden', 256),
    # FFN reduction
    ('ffn1600', 'ffn', 1600),
    ('ffn800',  'ffn',  800),
]


def export_onnx(model, path):
    """Export model to ONNX."""
    model._sample_latent = lambda mu, ls: mu  # deterministic
    torch.onnx.export(model, (torch.zeros(1,3,224,224), torch.zeros(1,2)),
        path, input_names=['images','state'], output_names=['action'],
        opset_version=14, dynamo=False, do_constant_folding=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='final_model.pt')
    parser.add_argument('--outdir', default='pruned_models')
    parser.add_argument('--skip-export', action='store_true')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    os.makedirs(args.outdir, exist_ok=True)

    # Baseline
    print("=== Baseline ===")
    b_model, b_cfg = prune_model(ckpt, 'enc', 4)[:2]
    with torch.no_grad():
        b_out = b_model(torch.zeros(1,3,224,224), torch.tensor([[0.5,-0.3]]), infer_cvae=True)['action'].squeeze().numpy()

    results = {}
    for name, strat, val in SINGLE_VARIANTS:
        print(f"\n=== {name} ({strat}={val}) ===")
        try:
            model, cfg, sd, missing, unexpected = prune_model(ckpt, strat, val)
            with torch.no_grad():
                out = model(torch.zeros(1,3,224,224), torch.tensor([[0.5,-0.3]]), infer_cvae=True)['action'].squeeze().numpy()
            diff = np.abs(out - b_out).max()
            params = sum(p.numel() for p in model.parameters())
            print(f"  Params: {params/1e6:.1f}M, Max diff: {diff:.6f}")
            print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            print(f"  First 4: {out[:4]}")

            if not args.skip_export:
                onnx_path = f'{args.outdir}/act_{name}.onnx'
                export_onnx(model, onnx_path)
                print(f"  ONNX: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f}MB)")

            results[name] = dict(strat=strat, val=val, max_diff=float(diff),
                                params_m=params/1e6, output=out.tolist())
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    with open(f'{args.outdir}/all_prune_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.outdir}/all_prune_results.json")
