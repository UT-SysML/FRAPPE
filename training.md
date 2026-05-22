# Training

The script `train_rae_progressive.py` at the repository root is the frozen
version used to produce the checkpoint reported in the paper. It imports the
model, ops, and quantizer definitions from `src/compressors/frappe/`, so it
must be run from the repository root.

## Example command

The example below trains a small 9-channel, 3-scale model with 1 epoch per
channel for each of the single-channel and merged-decoder phases. The
remaining hyperparameters use the script defaults (decoder_dim=768,
decoder_arch=CCCCCC, encoder_arch=SC8, LSDIR train / Kodak validation,
crop size 480, batch size 1).

```bash
python train_rae_progressive.py \
    --device cuda:0 \
    --ps 32 32 32 16 16 16 8 8 8 \
    --epochs_single 1 \
    --epochs_merged 1 \
    --save_checkpoint_name checkpoint_example.pth
```

This is much lighter than the full paper configuration
(`ps = [32×4, 16×3, 8×3, 4×3]`, `epochs_single=2`, `epochs_merged=4`,
plus a scaled-down `decoder_dim` and per-channel `lam` / `sc_max_lr`
schedule), and is intended for sanity-checking the pipeline rather than
matching reported rate-distortion numbers.

## What runs

For each of the 9 channels the script performs two phases:

1. **Single-channel residual training.** A new `AutoencoderSingleChannel`
   (one encoder filter at the current patch size + a full decoder) is
   trained to reconstruct the residual left by the currently merged
   model. Rate-distortion loss is `log10(MSE) + lam · target_power^rpe
   · log2(std(z))`.
2. **Merged-decoder retraining.** The new channel's encoder filter is
   merged into the `MergedAutoencoder`, all encoders are frozen and
   their latents quantized (`round()` to int8), and only the unified
   decoder is fine-tuned.

The decoder operates at `decoder_ps = max(ps) = 32` and finer latents
(ps=16, ps=8) are patchified up to that resolution before concatenation,
so the decoder input channel count grows from 1 (ch0) to 1+1+1+4+4+4+16+16+16
(ch8) without changing the decoder architecture.

After every channel, the script appends a snapshot to
`checkpoint_example.pth` and prints the validation PSNR and compression
ratio on Kodak. When all channels are done it computes BD-rate vs the
bundled AVIF reference points and writes a results JSON next to the
checkpoint.

## Resuming

Channel-level resumption is supported via `--resume_checkpoint` and
`--resume_channels N` (skips the first `N` channels and continues from
there). The new run's flags must match the saved config for the resumed
channels — `ps`, `decoder_ps`, `encoder_arch`, `decoder_arch`,
`decoder_dim`, `decoder_kernel_size`, `decoder_mlp_ratio`,
`decoder_layerscale`, and `input_channels` are all asserted at startup.

## Using the resulting checkpoint

The checkpoint stores per-channel-count snapshots in
`merged_decoder_weights[n]`. To rebuild a `MergedAutoencoder` for the
first `n+1` channels, instantiate it with the saved `config` and load
the encoder/decoder state dicts from the snapshot — the same pattern
used by `compressors.frappe.model.load_progressive_model` for
safetensors weights from the Hugging Face hub (see the inference
example).
