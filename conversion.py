import gradio as gr
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import numpy as np
import soundfile as sf 


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    # print(text_norm)
    return text_norm
hps = utils.get_hparams_from_file("./configs/vtubers.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
_ = net_g.eval()

# _ = utils.load_checkpoint("./G_49200.pth", net_g, None)
_ = utils.load_checkpoint("./nene_final.pth", net_g, None)
all_emotions = np.load("all_emotions.npy")
import random
def tts(txt, emotion):
    stn_tst = get_text(txt, hps)
    # print(stn_tst)
    randsample = None
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([1]) # speaker id
        emo = torch.FloatTensor(all_emotions[emotion]).unsqueeze(0)

        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1, emo=emo)[0][0,0].data.float().numpy()
    return audio, randsample


def tts1(text, emotion):
    if len(text) > 150:
        return "Error: Text is too long", None
    audio, _ = tts(text, emotion)
    # print(audio)
    return "Success", hps.data.sampling_rate, audio

msg, sr, audio = tts1("kimochi. like. input. macbook.",2)
# msg, sr, audio = tts1("私はプログラムするのが好きです",2)

sf.write('output.wav', audio, sr)