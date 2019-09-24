import librosa
import shutil
import argparse
import matplotlib.pyplot as plt
import math, pickle, os, glob
import numpy as np
from tqdm import tqdm
from utils import *
from utils.display import *
from utils.generic_utils import load_config
from utils.audio import AudioProcessor
from multiprocessing import Pool


def get_files(path, extension=".wav"):
    filenames = []
    for filename in glob.iglob(f"{path}/**/*{extension}", recursive=True):
        filenames += [filename]
    return filenames


def process_file(path):
    wav = ap.load_wav(path)
    mel = ap.melspectrogram(wav)
    if CONFIG.mode in ['mold', 'gauss']:
        # copy symbolic link of wav file
        quant = None
    elif type(CONFIG.mode) is int and CONFIG.mulaw:
        quant = ap.mulaw_encode(wav, CONFIG.mode)
        quant = quant.astype(np.int32)
    elif type(CONFIG.mode) is int:
        quant = ap.quantize(wav)
        quant = quant.clip(0, 2 ** CONFIG.mode - 1)
        quant = quant.astype(np.int32)
    return mel.astype(np.float32), quant, wav


def extract_feats(wav_path):
    idx = wav_path.split("/")[-1][:-4]
    try:
        m, quant, wav = process_file(wav_path)
    except:
        if args.ignore_errors:
            return None
        else:
            raise RuntimeError(" [!] Cannot process {}".format(wav_path))
    if quant is None and CONFIG.mode not in ['mold', 'gauss']:
        raise RuntimeError(" [!] Audio file cannot be quantized!")
    if not ( quant is None):
        assert quant.max() < 2 ** CONFIG.audio['bits'], wav_path
        assert quant.min() >= 0
        np.save(f"{QUANT_PATH}{idx}.npy", quant, allow_pickle=False)
    np.save(f"{MEL_PATH}{idx}.npy", m, allow_pickle=False)
    return idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="path to config file for feature extraction."
    )
    parser.add_argument(
        "--num_procs", type=int, default=4, help="number of parallel processes."
    )
    parser.add_argument(
        "--data_path", type=str, default='', help="data path to overwrite config.json."
    )
    parser.add_argument(
        "--out_path", type=str, default='', help="destination to write files."
    )
    parser.add_argument(
        "--ignore_errors", type=bool, default=False, help="ignore bad files."
    )
    
    args = parser.parse_args()

    config_path = args.config_path
    CONFIG = load_config(config_path)

    if args.data_path != '':
        CONFIG.data_path = args.data_path
    
    if type(CONFIG.mode) is int:
        CONFIG.audio['bits'] = CONFIG.mode
    ap = AudioProcessor(**CONFIG.audio)

    SEG_PATH = CONFIG.data_path
    # OUT_PATH = os.path.join(args.out_path, CONFIG.run_name, "data/")
    OUT_PATH = args.out_path
    QUANT_PATH = os.path.join(OUT_PATH, "quant/")
    MEL_PATH = os.path.join(OUT_PATH, "mel/")
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(QUANT_PATH, exist_ok=True)
    os.makedirs(MEL_PATH, exist_ok=True)

    wav_files = get_files(SEG_PATH)
    print(" > Number of audio files : {}".format(len(wav_files)))

    wav_file = wav_files[1]
    m, quant, wav = process_file(wav_file)

    # save an example for sanity check
    if type(CONFIG.mode) is int:
        wav_hat = ap.dequantize(quant)
        librosa.output.write_wav(
            OUT_PATH + "test_converted_audio.wav", wav_hat, sr=CONFIG.audio['sample_rate']
        )
        shutil.copyfile(wav_files[1], OUT_PATH + "test_target_audio.wav")

    # This will take a while depending on size of dataset
    with Pool(args.num_procs) as p:
        dataset_ids = list(tqdm(p.imap(extract_feats, wav_files), total=len(wav_files)))

    # remove None items
    if args.ignore_errors:
        dataset_ids = [idx for idx in dataset_ids if idx is not None]

    # save metadata
    with open(os.path.join(OUT_PATH, "dataset_ids.pkl"), "wb") as f:
        pickle.dump(dataset_ids, f)
