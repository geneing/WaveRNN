# coding: utf-8
"""
Preprocess dataset for WaveRNN

usage: preprocess.py [options] <in_dir> <out_dir>

options:
    <in_dir>                 folder containing your training wavs
    --num_workers=<n>        Num workers.
    --hparams=<parmas>       Hyper parameters [default: ].
    --preset=<json>          Path of preset parameters (json).
    -h, --help               Show help message.
"""
from docopt import docopt
from multiprocessing import cpu_count
from hparams import hparams
import pickle, os, glob
import numpy as np
from utils.dsp import DSP


def get_files(path, extension='.wav') :
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{extension}', recursive=True):
        filenames += [filename]
    return filenames


def convert_file(path):
    wav = dsp.load_wav(path, encode=False)
    mel = dsp.melspectrogram(wav)
    quant = (wav + 1.) * (2**hparams.bits - 1) / 2
    return mel.astype(np.float32), quant.astype(np.int)


if __name__ == "__main__":
    args = docopt(__doc__)

    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    num_workers = args["--num_workers"]
    num_workers = cpu_count() if num_workers is None else int(num_workers)
    preset = args["--preset"]
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "WaveRNN"

    dsp = DSP(hparams)
    quant_path = os.path.join(out_dir, 'quant/')
    mel_path = os.path.join(out_dir, 'mel/')
    os.makedirs(quant_path, exist_ok=True)
    os.makedirs(mel_path, exist_ok=True)

    wav_files = get_files(in_dir)

    # This will take a while depending on size of dataset
    dataset_ids = []
    for i, path in enumerate(wav_files) :
        dataset_id = path.split('/')[-1][:-4]
        dataset_ids += [dataset_id]
        m, x = convert_file(path)
        np.save(f'{mel_path}{dataset_id}.npy', m)
        np.save(f'{quant_path}{dataset_id}.npy', x)
        print('%i/%i'%(i + 1, len(wav_files)))
    with open(os.path.join(out_dir,'dataset_ids.pkl'), 'wb') as f:
        pickle.dump(dataset_ids, f)