setup for QML people: 
1. git clone https://github.com/l3mxny/QML.git
2. cd QML
3. python3 -m venv qugen_env
4. source qugen_env/bin/activate
5. pip install .
6. pip install jupyter

To see results: 
1. cd QML
2. source qugen_env/bin/activate
3. jupyter notebook




[![arXiv](https://img.shields.io/badge/arXiv-2603.00233-fb595a.svg)](https://arxiv.org/abs/2603.00233)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18791062.svg)](https://doi.org/10.5281/zenodo.18791062)

This repository contains the supplementary codebase and data for the paper:

Jäger, Kiwit, and Riofrío, **Scaling Quantum Machine Learning without Tricks: High-Resolution and Diverse Image Generation**, 2026,

available as a preprint on the [arXiv](https://arxiv.org/abs/2603.00233).

The quantum generative adversarial network (QGAN) implementation is built upon
the [qugen](https://github.com/QutacQuantum/qugen) Python library. The baseline code was imported from commit `68fda95`. We thank the original authors for their open-source contributions.

# Organization of the repository

The repository is organized as follows:

- Training data sets must be stored and preprocessed in ''apps/logistics/training_data'' (scripts to preprocess raw data
  sets are included).
- Scripts to train and evaluate models are provided in ''run_scripts''.
- Any new experiment results are automatically stored in the ''apps/logistics/experiments'' folder.
- Model files of the trained QGANs are stored in the ''experiments'' folder.
- The corresponding QGAN training/experiment configuration files are provided in ''experiments_config'' (in CSV format).
  Each line corresponds to a single experiment/model.
- All figures and results presented in the paper are created in the Jupyter notebook ''plots_paper.ipynb''.
- The main implementation of the QGANs, based on the underlying qugen library, is found in ''qugen/main''.
- Export and execution of trained generators on IBM Quantum hardware is in ''qugen/main/hardware''.
- Analysis scripts for the shot-count and small-circuit studies are in ''run_scripts/analysis'', and the digit
  recognizability/coverage script is ''run_scripts/digit_coverage.py''.
- Unit tests are in ''qugen/test''.

# Installation

The pinned JAX version (0.6.0) requires Python 3.10 or later.

1) Create and activate a virtual environment, either with conda:
   ``conda create --name qugen_env python=3.12.9`` followed by ``conda activate qugen_env``,
   or with the built-in venv module (macOS/Linux): ``python3 -m venv qugen_env`` followed by ``source qugen_env/bin/activate``.
2) Upgrade pip. Older pip versions cannot find the pinned JAX release: ``python -m pip install --upgrade pip``
3) From the repository root, run ``pip install .`` or ``pip install -e .`` to install it in editable mode.
   To use the Jupyter notebooks, also run ``pip install jupyter``.
4) *(optional)* For IBM Quantum hardware and Qiskit simulators, run ``pip install ".[ibm]"``.

### CUDA installation instructions

In order to run the models on CUDA GPUs, a CUDA supporting JAX built must be installed.
Typically, when installing the packages as instructed above, JAX is installed with CPU support only.
For CUDA support, perform the following steps (we recommend to create a new environment to switch between CPU and GPU
computing)

1) *(optional, recommended)* Create a new virtual environment (with a different name), activate it and install the
   requirements as specified in the three-step installation guide above.
2) Uninstall JAX via ``pip uninstall jax``
3) Determine CUDA driver version, e.g. using ``nvidia-smi``
4) Install JAX with the matching CUDA version, e.g. using ``pip install "jax[cuda12]==0.6.0"`` for CUDA driver version
    12. More information is provided in
        the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-via-pip-easier).

# Usage

## Training QGAN models

To train QGAN models, use the script ''run_scripts/run_experiments.py'' as follows:

```
cd apps/logistics
python ../../run_scripts/run_experiments.py --experiments_csv_path=[experiments_config.csv]
```

The argument ``experiments_csv_path`` specifies the path to the experiments configuration file (in CSV format) that
specifies the training settings for one or multiple experiments (one experiment per line).
Importantly, the first column specifies the result path and must be kept blank for experiments that are meant to be
trained.
When the training is completed successfully, this column is automatically updated to indicate the path where the model
and training results are stored.
All such configuration files used for the experiments presented in the paper are provided in the folder ''
experiments_configs''.
The training results are stored in a new model folder in ''apps/logistics/experiments'' indicated in the experiment
configuration file.
Note that the training datasets must be stored in ''apps/logistics/training_data'' and appropriately preprocessed (see
below).
(Multi-CPU/GPU training is supported and can be activated by the CLI flag ``n_processes``.)

## Shot noise and random seeds

By default (``shots=None``) the generator's output probabilities are exact. Setting ``shots`` (e.g. 128, 512 or 2048)
samples a finite number of measurements from those probabilities, simulating a real quantum device. ``shots`` can be
passed to ``train_image_qgan.py`` or set as a column in the experiments CSV. Probabilities are clipped away from zero
before sampling so that shot-based runs do not produce NaNs.

Randomness is controlled as follows:

- ``random_seed`` (default 42) in ``ContinuousQGANModelHandler`` seeds the JAX random key used for latent noise,
  gradient-penalty interpolation and shot sampling. It is not exposed as a command-line argument of
  ``train_image_qgan.py``.
- The key is split at every training step, so the latent noise and the shot noise change from one iteration to the next.
  Reusing a fixed key froze the shot noise, so the model could learn one fixed noise pattern.
- Loading a saved model resets its key to a fixed seed (2), so sampling from a loaded model is reproducible.

## Evaluation of trained QGAN models

### Fréchet Inception Distance (FID)

To run the Fréchet Inception Distance (FID) evaluation of the trained models reported in the paper, run the Jupyter notebook ''FID_evaluation.ipynb''.
Please note that for some models, precomputed samples are used. These are created when running the Jupyter notebook ''plots_paper.ipynb'', which, hence, should be run before performing the FID evaluation. As the FIDs are evaluated with respect to the real datasets, these datasets have to be properly preprocessed and stored in ''apps/logistics/training_data'' (see section below on _Training dataset preparation_). The following datasets are used (classes for subsets are indicated in braces): MNIST, MNIST{0,1,2}, Fashion-MNIST, Fashion-MNIST{0,1}, SVHN{0}. Furthermore, to include the FID evaluation of the Patch-QGAN models (Tsang et al., 2023), please run the Jupyter notebook ''patchQGAN_benchmark.ipynb'' first to ensure that the samples are generated. 

### Maximum Mean Discrepancy (MMD)

To run the Maximum Mean Discrepancy (MMD) evaluation of trained models as used for the results presented in the paper,
use the script ''run_scripts/run_evaluations.py'' as follows:

```
cd apps/logistics
python ../../run_scripts/run_evaluations.py --experiments_csv_path=[experiments_config.csv] --eval_config_csv_path="../../experiments_configs/mmd_eval.csv" --n_iters_skip=[N] --n_samples=[M]
```

The argument ``experiments_csv_path`` specifies the path to the experiments configuration file (in CSV format) that was
used to train the models, such as for the trained models in ``experiments_configs``.
Further, specify the number N of iterations to skip between evaluating a checkpointed model (e.g. N=499 iterations used
for the paper results) via ``n_iters_skip``. Note that by default checkpoints are only created every 100 iterations, N
should be chosen as a multiple of 100 minus 1 to hit the checkpoints.
Provide the number M of samples to generate from the trained model to
evaluate the MMD metrics (M=5000 in the paper) via ``n_samples``.
The evaluation results are stored in the model folder indicated in the experiment configuration file and named ''
evaluation_summary.csv''.

### Digit recognizability and class coverage

``run_scripts/digit_coverage.py`` fits a small classifier on labelled real images (preprocess them with
``--save_labels=True``). It reports the classifier's held-out accuracy on real images at the chosen resolution, and the
predicted digit histogram of generated images, which shows whether all ten digits are produced. Run it from
``apps/logistics``:

```
python ../../run_scripts/digit_coverage.py --data_set_name=mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area
python ../../run_scripts/digit_coverage.py --data_set_name=mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area --model_name=[model_folder_name] --n_samples=1000
```

Generated images can also be read from a saved ``.npy`` file with ``--samples_path``.

## Plots

The Jupyter notebook ''plots_paper.ipynb'' creates all plots and figures presented in the paper.
For example, it loads images stored during training or generates new images from trained models (based on parameter
checkpoints), which may take a while.
The final plots are automatically stored in the folder ''plots_paper''.

## Training dataset preparation

Training datasets must be stored in the ''apps/logistics/training_data'' folder in numpy format (``.npy`` files).
Scripts to preprocess raw datasets are provided in the same folder, i.e., ''mnist_preprocessing.py'', ''
fashion_mnist_preprocessing.py'', and ''svhn_preprocessing.py''.
Each of these scripts specifies in comments where to download the raw dataset and how to name the files when placing
them in the ''apps/logistics/training_data'' folder. For convenience, these instructions are repeated below.
Importantly, the QGAN implementation expects image resolutions of powers of two, e.g. 32x32, which must be specified
when running the preprocessing scripts.
Furthermore, to create a training dataset with only a subset of classes, the desired classes can be specified as a
comma-separated list of integers.
As an example, to preprocess the MNIST dataset to 32x32 resolution for the digits 0 to 2, run

```
cd apps/logistics/training_data
python mnist_processing.py --img_size=32 --digit="[0,1,2]"
```

Two further options of ``mnist_processing.py``:

- ``--interpolation`` selects the OpenCV downscaling method (``lanczos``, the default used for the paper's experiments,
  or ``area``, ``linear``, ``cubic`` and ``nearest``). OpenCV recommends ``area`` for strong downscaling, e.g. 28x28 to 4x4.
- ``--save_labels=True`` also saves the digit labels, which ``digit_coverage.py`` needs.

The following raw datasets are required to be downloaded manually and placed in the ''apps/logistics/training_data''
folder before running the respective preprocessing scripts:

- MNIST:  Download raw data (''train_images.npy'' and ''
  train_labels.npy'') [here](https://github.com/sebastian-lapuschkin/lrp_toolbox/tree/master/data/MNIST) and save files
  as ''mnist_raw.npy'' and ''mnist_labels_raw.npy'', respectively.
- Fashion-MNIST: Download raw data (''train-images-idx3-ubyte.gz'' and ''
  train-labels-idx1-ubyte.gz'') [here](https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file) and save
  files as ''fashion_mnist_train_raw.gz'' and ''fashion_mnist_train_labels_raw.gz'', respectively.
- SVHN: Download raw data (''extra_32x32.mat'') [here](http://ufldl.stanford.edu/housenumbers/) and save the file as ''
  svhn_extra_32x32.mat''.

# Running generators on IBM Quantum hardware

The modules in ''qugen/main/hardware'' run trained generators on IBM Quantum backends:

- ''qiskit_export.py'' converts a trained generator into Qiskit circuits, one per noise sample. The measured counts are
  decoded in the same way as the simulated probabilities during training. ''counts_to_probs()'' converts between the bit
  ordering of PennyLane and Qiskit.
- ''ibm.py'' transpiles and runs the circuits through Qiskit Runtime. Backend names are ''aer'' (ideal simulation with
  shots), ''fake_<name>'' (noisy simulation of a device snapshot, e.g. ''fake_torino'', which needs no account),
  ''least_busy'' (the least busy device of your account) or a device name such as ''ibm_torino''.

Real devices require an IBM Quantum account saved beforehand with ''QiskitRuntimeService.save_account(...)''. The
repository does not store credentials. Install the dependencies with ''pip install ".[ibm]"''.

# Analysis scripts

''run_scripts/analysis'' contains the scripts for the shot-count and small-circuit studies:

- *Phase 1* (''small_circuit_sweep.py''): trains a grid of small circuit configurations for a short, fixed budget and logs
  KL divergence and parameter counts. It is designed for Colab or Kaggle rather than local machines.
- *Phase 2* (''shot_count_sweep.py''): sweeps the shot count, both on an already trained generator and during training
  from scratch. Example: ''python run_scripts/analysis/shot_count_sweep.py --n_qubits=7 --circuit_depth=2''
- *Phase 4* (''synthesize_report.py''): joins the outputs of the earlier phases into one ranked table and a short summary.
- ''circuit_resource_report.py'' reports resource specifications of circuit configurations, ''toy_data.py'' generates
  the bars-and-stripes data used by the sweeps, and ''shot_noise_fix_demo.ipynb'' demonstrates the per-step random key.

# Tests

Run the unit tests from the repository root with ''pytest qugen/test''. The tests in ''qugen/test/hardware'' require the
optional IBM dependencies (''pip install ".[ibm]"'').

# Miscellaneous

#### Config files

A convenient way to specify multiple trainings is offered by creating config files in the CSV format and should be
stored in ''apps/logistics/experiments''.
Each row specifies a single experiment setting, while each column is linked to a ``main`` function argument in ''
run_scripts/train_image_qgan.py'' (default values are used if not specified) and therefore has to match these
argument names. Alternatively, when using the CLI of ''run_scripts/run_experiments.py'', provide this config file
path via the parameter ``--experiments_csv_path={}``
However, the following exceptions exist:

- *result_path*: This column/value is automatically organized by ''run_scripts/run_experiments.py'' and can be in
  five different states

  | Value                      | Description                                                                                                                                                                                                                                                                            |
  |----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | *(Empty)*                  | This experiment has not been run. In order to run this experiment the next time when the config file is passed to ''run_scripts/run_experiments.py'', this cell should left empty. (note that in CSV this is realized by directly providing the next comma without any whitespace.) |
  | *...*                      | This experiment is currently queued for training by a ''run_scripts/run_experiments.py'' run.                                                                                                                                                                                       |
  | *...(p{id})*               | This experiment is currently trained and executed by the process with ID *id*.                                                                                                                                                                                                         |
  | *...ERR:{msg}*             | The training of this experiment raised an error with message *msg* and is ended.                                                                                                                                                                                                       |
  | *experiments/{model_name}* | The experiment has been completed successfully and the results can be found under the specified path.                                                                                                                                                                                  |

- *data_set_{}*: The argument ``data_set_name`` must either correspond to a numpy training data set file in ''
  apps/logistics/training_data'' or it can be a Python f-string, i.e., contains unspecified keywords indicated by curly
  braces. In the latter case, all other arguments/columns starting with *data_set_* are aggregated as data set keyword
  arguments and used to construct the final ``data_set_name`` by replacing the keywords accordingly.
  For example, ``data_set_name = mnist{data_set_class}{data_set_img_size}x{data_set_img_size}_N_{data_set_n_samples}``
  with values *_0_1_2_3_4_5_6_7_8_9_*, *32*, and *60000* for the keywords *data_set_class*, *data_set_img_size*, and
  *data_set_n_samples*, respectively, is turned into ``data_set_name = mnist_0_1_2_3_4_5_6_7_8_9_32x32_N_60000``.

An example for a configuration CSV file specifying three experiments (one completed, one in progress, one unscheduled)
is provided here:

| result_path                                                                         | data_set_name                                                                       | data_set_class        | data_set_img_size | data_set_n_samples | model_name | circuit_depth | noise_distr       | noise_scale | noise_shift                                                                                                   | init_noise_distr | init_noise_scale | init_noise_shift | transformation | measurement_scheme | decoding_scheme | save_artifacts                             | discriminator_name | generator_name                                             | reupload | gen_init_distr | gen_init_scale | single_data_point | n_epochs | gan_method | noise_tuning | initial_learning_rate_generator | initial_learning_rate_discriminator | discriminator_training_steps | adam_b1 | adam_b2 | batch_size | n_ancilla_qubits | warm_start | version |
|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-----------------------|-------------------|--------------------|------------|---------------|-------------------|-------------|---------------------------------------------------------------------------------------------------------------|------------------|------------------|------------------|----------------|--------------------|-----------------|--------------------------------------------|--------------------|------------------------------------------------------------|----------|----------------|----------------|-------------------|----------|------------|--------------|---------------------------------|-------------------------------------|------------------------------|---------|---------|------------|------------------|------------|---------|
| experiments/continuous_mnist_0_1_2_3_4_5_6_7_8_9_32x32_N_60000_minmax_qgan_3bf2e3a4 | mnist{data_set_class}{data_set_img_size}x{data_set_img_size}_N_{data_set_n_samples} | _0_1_2_3_4_5_6_7_8_9_ | 32                | 60000              | continuous | (32, 3)       | normal_multi_mode | 0.1         | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 |                  |                  |                  | minmax         | comp_basis_probs   | FRQI_msb        | 'samples_20','noise','measurement_outputs' | convolutional      | color_rot_skip_SO4_blocks_noise_angle_next_neighbor_mirror | 0        | normal         | 0.01           |                   | 50000    | WGAN_10    | 40           | 0.001                           | 0.0001                              | 10                           | 0.5     | 0.9     | 64         | 0                | 0          | 1.0     |
| ...(p424242)                                                                        | mnist{data_set_class}{data_set_img_size}x{data_set_img_size}_N_{data_set_n_samples} | _0_1_2_3_4_5_6_7_8_9_ | 32                | 60000              | continuous | (64, 4)       | normal_multi_mode | 0.1         | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 |                  |                  |                  | minmax         | comp_basis_probs   | FRQI_msb        | 'samples_20','noise','measurement_outputs' | convolutional      | color_rot_skip_SO4_blocks_noise_angle_next_neighbor_mirror | 0        | normal         | 0.01           |                   | 50000    | WGAN_10    | 40           | 0.001                           | 0.0001                              | 10                           | 0.5     | 0.9     | 64         | 0                | 0          | 1.0     |
|                                                                                     | mnist{data_set_class}{data_set_img_size}x{data_set_img_size}_N_{data_set_n_samples} | _0_1_2_3_4_5_6_7_8_9_ | 32                | 60000              | continuous | (72, 3)       | normal_multi_mode | 0.1         | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 |                  |                  |                  | minmax         | comp_basis_probs   | FRQI_msb        | 'samples_20','noise','measurement_outputs' | convolutional      | color_rot_skip_SO4_blocks_noise_angle_next_neighbor_mirror | 0        | normal         | 0.01           |                   | 50000    | WGAN_10    | 40           | 0.001                           | 0.0001                              | 10                           | 0.5     | 0.9     | 64         | 0                | 0          | 1.0     |

#### Parallel execution of experiments in config file

For parallel execution of *N* experiments provided in a config file, increase the argument ``n_processes`` to *N* in the
``main`` function in ''run_scripts/run_experiments.py'' (or pass the parameter ``--n_processes=N`` via the CLI of ''
run_scripts/run_experiments.py'').
When run in GPU mode, this will reserve a single GPU per experiment. The least busy GPUs (by memory utilization) are
automatically determined and the top-level list variable ``EXCLUDE_GPUS`` in ''run_scripts/run_experiments.py'' can
be used to exclude certain GPUs by their ID from the training. 

#### Patch-QGAN benchmarking

To run the benchmark with the Patch-QGAN framework (Tsang et al., 2023), which reproduces the models and generates samples, please refer to the Jupyter notebook ''patchQGAN_benchmark.ipynb''. This is based on the original implementation provided on [GitHub](https://github.com/jasontslxd/PQWGAN). This repository is automatically cloned when running the notebook. The model checkpoint files for the trained models reproduced for the present work are provided in ''experiments/patch_qgan_models''.