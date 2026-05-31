"""
ACT model pruning: layer pruning + head pruning.
Exports ONNX for each variant.
"""
import torch
import numpy as np
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from act.modeling_act import ACTModel
from act.configuration_act import ACTConfig


def create_pruned_model(ckpt_path, enc_layers, dec_layers, num_heads=None):
    """Create a pruned model by reducing encoder/decoder layers and/or attention heads."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg_dict = ckpt['config']
    sd = ckpt['model_state_dict']

    # Build new config with reduced layers/heads
    new_cfg = ACTConfig(
        state_dim=cfg_dict['state_dim'],
        action_dim=cfg_dict['action_dim'],
        action_chunk_size=cfg_dict['action_chunk_size'],
        n_action_steps=cfg_dict['action_chunk_size'],
        hidden_dim=cfg_dict['hidden_dim'],
        num_attention_heads=num_heads if num_heads else cfg_dict['num_attention_heads'],
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dim_feedforward=cfg_dict['dim_feedforward'],
        latent_dim=cfg_dict['latent_dim'],
        use_cvae=cfg_dict['use_cvae'],
        kl_weight=cfg_dict['kl_weight'],
        use_temporal_ensembling=False,
    )

    model = ACTModel(new_cfg)
    model.eval()

    # Set deterministic latent
    model.set_inference_latent(ckpt['inference_latent_mu'], ckpt['inference_latent_log_sigma'])

    # Load matching weights (strict=False drops pruned layers/heads)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Loaded: {len(sd)} keys, missing={len(missing)}, unexpected={len(unexpected)}")

    return model, new_cfg


def export_onnx(model, output_path, opset=14):
    """Export model to ONNX with deterministic CVAE."""
    images = torch.zeros(1, 3, 224, 224)
    state = torch.zeros(1, 2)

    # Monkey-patch for deterministic CVAE (eps=0, z=mu)
    original_sample = model._sample_latent
    model._sample_latent = lambda mu, log_sigma_x2: mu

    torch.onnx.export(
        model,
        (images, state),
        output_path,
        input_names=['images', 'state'],
        output_names=['action'],
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
    )

    # Restore
    model._sample_latent = original_sample
    return output_path


def test_inference(model, images=None, state_val=None):
    """Run a quick inference test."""
    if images is None:
        images = torch.zeros(1, 3, 224, 224)
    if state_val is None:
        state_val = torch.tensor([[0.5, -0.3]])
    with torch.no_grad():
        out = model(images, state_val, infer_cvae=True)['action']
    return out.squeeze().numpy()


VARIANTS = {
    # Encoder pruning (decoder intact — decoder is critical)
    'E3D4':  (3, 4, None),  # drop 1 enc
    'E2D4':  (2, 4, None),  # drop 2 enc
    'E1D4':  (1, 4, None),  # drop 3 enc (aggressive)
    # Decoder pruning (encoder intact — bad idea, for reference)
    'E4D3':  (4, 3, None),  # drop 1 dec
    'E4D2':  (4, 2, None),  # drop 2 dec
    # Both
    'E3D3':  (3, 3, None),
    'E2D2':  (2, 2, None),
    # Head pruning (reference only, no ONNX size benefit on TPU)
    'H4':    (4, 4, 4),
    'H2':    (4, 4, 2),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='final_model.pt')
    parser.add_argument('--outdir', default='pruned_models')
    parser.add_argument('--variant', default=None, help='Specific variant or "all"')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    variants = {args.variant: VARIANTS[args.variant]} if args.variant and args.variant != 'all' else VARIANTS

    # Baseline (unpruned)
    print("=== Baseline (unpruned) ===")
    model, cfg = create_pruned_model(args.ckpt, 4, 4, 8)
    baseline_out = test_inference(model)
    print(f"  Output[:4]: {baseline_out[:4]}")

    results = {}
    for name, (enc, dec, heads) in variants.items():
        print(f"\n=== Variant: {name} (E{enc}D{dec}" + (f"H{heads})" if heads else ")"))
        try:
            model, cfg = create_pruned_model(args.ckpt, enc, dec, heads)
            out = test_inference(model)
            diff = np.abs(out - baseline_out).max()
            print(f"  Output[:4]: {out[:4]}")
            print(f"  Max diff vs baseline: {diff:.6f}")

            onnx_path = os.path.join(args.outdir, f'act_{name}.onnx')
            export_onnx(model, onnx_path)
            print(f"  ONNX: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")

            results[name] = {
                'enc_layers': enc, 'dec_layers': dec, 'heads': heads,
                'max_diff': float(diff),
                'onnx': onnx_path,
                'output': out.tolist(),
            }
        except Exception as e:
            print(f"  FAILED: {e}")

    # Save report
    report_path = os.path.join(args.outdir, 'prune_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nReport: {report_path}")
