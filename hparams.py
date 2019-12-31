
import tensorflow as tf

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    name="WaveRNN",
    batch_size=64,
    batch_size_gen=32,
    num_workers=8,
    rnn_dims=512,
    fc_dims=512,
    pad=2,
    upsample_factors=(5, 5, 11),
    feat_dims=80,
    compute_dims=128,
    res_out_dims=128,
    res_blocks=10,
    epochs=1000,
    lr=1.e-4,
    orig_sample_rate=22050,
    sample_rate=22050,
    n_fft=2048,
    num_mels=80,
    hop_period=0.0125,  # 12.5ms
    win_period=0.05,  # 50ms
    fmin=40,
    min_level_db=-100,
    ref_level_db=20,
    bits=10
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)