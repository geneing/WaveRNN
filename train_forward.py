import torch
import traceback
from torch import optim, nn
import torch.nn.functional as F

from models.forward_tacotron import ForwardTacotron
from utils import hparams as hp
from utils.dataset import get_tts_datasets
from utils.display import *
from utils.text.symbols import symbols
from utils.paths import Paths
from models.tacotron import Tacotron
import argparse
from utils import data_parallel_workaround
from pathlib import Path
import time
import numpy as np
from utils.checkpoints import save_checkpoint, restore_checkpoint


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    force_gta = args.force_gta

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        for session in hp.forward_schedule:
            _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate Forward TTS Model
    print('\nInitialising Forward TTS Model...\n')
    model = ForwardTacotron(embed_dims=hp.forward_embed_dims,
                            num_chars=len(symbols),
                            durpred_rnn_dims=hp.forward_durpred_rnn_dims,
                            durpred_conv_dims=hp.forward_durpred_conv_dims,
                            rnn_dim=hp.forward_rnn_dims,
                            postnet_k=hp.forward_postnet_K,
                            postnet_dims=hp.forward_postnet_dims,
                            prenet_k=hp.forward_prenet_K,
                            prenet_dims=hp.forward_prenet_dims,
                            highways=hp.forward_num_highways,
                            dropout=hp.forward_dropout,
                            n_mels=hp.num_mels).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'num params {params}')

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint('forward', paths, model, optimizer, create_if_missing=True)

    if not force_gta:
        for i, session in enumerate(hp.forward_schedule):
            current_step = model.get_step()

            lr, max_step, batch_size = session

            training_steps = max_step - current_step

            simple_table([(f'Steps', str(training_steps//1000) + 'k Steps'),
                          ('Batch Size', batch_size),
                          ('Learning Rate', lr)])

            train_set, mel_example = get_tts_datasets(paths.data, batch_size, 1, alignments=True)
            train_loop(paths, model, optimizer, train_set, lr, training_steps, mel_example)

    train_set, mel_example = get_tts_datasets(paths.data, 8, 1, alignments=True)
    create_gta_features(model, train_set, paths.gta)
    print('Training Complete.')


def train_loop(paths: Paths, model, optimizer, train_set, lr, train_steps, mel_example):
    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    for e in range(1, epochs+1):

        start = time.time()
        running_loss = 0
        dur_running_loss = 0

        # Perform 1 epoch
        for i, (x, m, ids, mel_len, dur) in enumerate(train_set, 1):

            x, m, dur = x.to(device), m.to(device), dur.to(device)
            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                m_hat, m_post_hat, dur_hat = data_parallel_workaround(model, x, m, dur)
            else:
                m_hat, m_post_hat, dur_hat = model(x, m, dur)

            lin_loss = F.l1_loss(m_hat, m)
            post_loss = F.l1_loss(m_post_hat, m)
            dur_loss = F.l1_loss(dur_hat, dur)
            loss = lin_loss + post_loss + dur_loss
            optimizer.zero_grad()

            loss.backward()

            if hp.forward_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.forward_clip_grad_norm)
                if np.isnan(grad_norm):
                    print('grad_norm was NaN!')

            optimizer.step()

            running_loss += post_loss.item()
            avg_loss = running_loss / i
            dur_running_loss += dur_loss.item()
            dur_avg_loss = dur_running_loss / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            if step % hp.forward_checkpoint_every == 0:
                ckpt_name = f'fast_speech_step{k}K'
                save_checkpoint('forward', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)

            if mel_example in ids:
                idx = ids.index(mel_example)
                try:
                    seq = x[idx].tolist()
                    m_gen = model.generate(seq)
                    save_spectrogram(m_gen, paths.forward_mel_plot / f'{step}_gen', 600)
                except Exception:
                    traceback.print_exc()
                save_spectrogram(np_now(m_post_hat[idx]), paths.forward_mel_plot/f'{step}_gta', 600)
                save_spectrogram(np_now(m[idx]), paths.forward_mel_plot/f'{step}_target', 600)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Mel Loss: {avg_loss:#.4} ' \
                  f'| Duration Loss: {dur_avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)
        model.log(paths.forward_log, msg)

        save_checkpoint('forward', paths, model, optimizer, is_silent=True)


def create_gta_features(model: Tacotron, train_set, save_path: Path):
    device = next(model.parameters()).device  # use same device as model parameters

    iters = len(train_set)

    for i, (x, mels, ids, mel_lens, dur) in enumerate(train_set, 1):

        x, mels, dur = x.to(device), mels.to(device), dur.to(device)

        with torch.no_grad(): _, gta, _ = model(x, mels, dur)

        gta = gta.cpu().numpy()

        for j, item_id in enumerate(ids):
            mel = gta[j][:, :mel_lens[j]]
            mel = (mel + 4) / 8
            np.save(save_path/f'{item_id}.npy', mel, allow_pickle=False)

        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


if __name__ == "__main__":
    main()
