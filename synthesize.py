"""Synthesis script for WaveRNN vocoder

usage: synthesize.py [options] <mel_input.npy>

options:
    --checkpoint-dir=<dir>       Directory where model checkpoint is saved [default: checkpoints].
    --output-dir=<dir>           Output Directory [default: model_outputs]
    --hparams=<params>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --speaker-id=<N>             Use specific speaker of data in case for multi-speaker datasets.
    --no-cuda                    Don't run on GPU
    -h, --help                   Show this help message and exit
"""
import os
import librosa

from docopt import docopt
from utils.dsp import DSP
from models import *



if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    output_path = args["--output-dir"]
    checkpoint_path = args["--checkpoint"]
    preset = args["--preset"]
    no_cuda = args["--no-cuda"]

    device = torch.device("cpu" if no_cuda else "cuda")

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "WaveRNN"

    mel_file_name = args['<mel_input.npy>']

    dsp = DSP(hparams)

    model_path = f'{checkpoint_dir}/model.pyt'
    checkpoint_step_path = f'{checkpoint_dir}/model_step.npy'
    os.makedirs(output_path, exist_ok=True)

    model = Model(device=device, rnn_dims=hparams.rnn_dims, fc_dims=hparams.fc_dims, bits=hparams.bits, pad=hparams.pad,
                  upsample_factors=hparams.upsample_factors, feat_dims=hparams.feat_dims,
                  compute_dims=hparams.compute_dims, res_out_dims=hparams.res_out_dims, res_blocks=hparams.res_blocks).to(device)

    model.load_state_dict(torch.load(model_path), strict=False)

    mel = np.load(mel_file_name)
    output = model.generate(mel)
    librosa.output.write_wav(os.path.join(output_path, os.path.basename(mel_file_name)+'.wav'), output, hparams.sample_rate)


