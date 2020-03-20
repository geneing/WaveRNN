import os
import argparse
import torch
import importlib
import numpy as np
import scipy
import re

#from nltk.tokenize import sent_tokenize
#from spacy.lang.en import English

import nltk
from nltk.corpus.reader.util import *
from notebook_utils.synthesize import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input text file.',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save final wav file.',
    )


    args = parser.parse_args()

    init_hparams('notebook_utils/pretrained_hparams.py')
    tts_model = get_forward_model('pretrained/forward_100K.pyt')
    voc_model = get_wavernn_model('pretrained/wave_800K.pyt')

    try:
        path = os.path.realpath(os.path.dirname(__file__))
    except NameError as e:
        path = './'

    with open(args.input,'r') as f:
        txt = f.read()
    cleared_txt = re.sub('\n{2,}', '\n\n', txt)
    paragraphs = cleared_txt.split('\n\n')

    for i, p in enumerate(paragraphs):
        input_text = p.replace('\n', ' ')
        wav = synthesize(input_text, tts_model, voc_model, alpha=1.0)
        scipy.io.wavfile.write('%s/%d.wav'%(args.output, i+1), hp.sample_rate, wav.astype(np.float32))
