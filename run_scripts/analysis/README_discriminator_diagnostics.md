# Discriminator diagnostics

Two scripts for checking the classical critic (`qugen/main/discriminator/cnn_discriminator.py`)
during and after WGAN-GP training, plus one small logging addition in the training handler.

## Background

Only `ConvDiscriminator` is used in the paper's experiments: every config sets
`discriminator_name = convolutional`, and `factory.py` maps that string to the class. It is a
classical CNN (three 5x5 stride-2 convolutions, 64/128/256 filters, then one dense output,
about 1.03M parameters). With `is_critic=True` the final sigmoid is skipped, so the output is an
open-ended Wasserstein score rather than a probability.

Training is driven from `qugen/main/generator/continuous_qgan_model_handler.py`. `gan_method =
WGAN_10` means WGAN with gradient-penalty coefficient 10.

## 1. Loss and gradient curves

```bash
python run_scripts/analysis/discriminator_diagnostics.py \
    --experiments-dir experiments \
    --out-dir plots_discriminator \
    --per-run-plots            # optional, one figure per run
```

Reads every `evaluation_summary.csv`, writes an overview figure, optional per-run figures, and
`discriminator_run_summary.csv` with start/end values for each run.

Two caveats apply to the logged `loss_discriminator` column and are repeated in the captions:

1. it includes the gradient-penalty term, so it is not the pure Wasserstein estimate, and
2. only the last of the `discriminator_training_steps` inner updates is recorded per iteration,
   so it is a snapshot rather than an average.

Expected pattern: the critic loss is negative and climbs towards zero as the generator improves,
because the loss is roughly the negative gap between real and fake scores. Its magnitude scales
with image size, so values are comparable within a resolution, not across resolutions.

## 2. Real vs generated scores, and the gradient-penalty check

```bash
python run_scripts/analysis/discriminator_real_vs_fake.py \
    --run experiments/<model folder> \
    --real-data apps/logistics/training_data/<dataset>.npy \
    --iteration 20000 \
    --out-dir plots_discriminator
```

By default the fake images come from the `samples_iteration=*.npy` file saved during training,
which needs no quantum simulation. Pass `--generate` to re-run the generator instead (slower,
needs the full training dependencies, but gives as many samples as you ask for). Use
`--n-channels 3` for SVHN.

The script reports:

* mean critic score for real and generated images, and the gap between them. Under the WGAN
  formulation the critic is *meant* to separate the two, and that gap is its Wasserstein estimate,
  so the useful signal is how the gap shrinks across checkpoints, not the separation itself;
* a split of the two score piles at the midpoint. The threshold comes from the same scores being
  measured, so treat it as descriptive, not as a held-out classifier accuracy;
* `||grad_x D||` on interpolated real/fake images, which the penalty constrains towards 1;
* the implied `-gap + penalty`, which should match the logged `loss_discriminator` at that
  checkpoint. On the 32-layer Fashion-MNIST model at iteration 20000 this reproduced the logged
  value to three decimals, confirming the logged loss is the Wasserstein gap plus the penalty.

## 3. Logging `||grad_x D||` during training

`slopes` is computed inside `cost_fn_discriminator` and discarded, so the penalty constraint could
not be checked from the training logs. The handler now records the mean value every
`slope_log_every` generator updates (default 100, `0` disables it) into a `grad_norm_interpolates`
column, with `NaN` on the iterations in between so the column stays aligned with the others.

It is diagnostic only: the value never feeds back into training, and the extra cost is one
generator forward pass per recorded iteration. `discriminator_diagnostics.py` picks the column up
automatically when it is present.
