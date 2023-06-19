import os
import gradio as gr
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import numpy as np
import soundfile as sf 
import librosa
import random
import pandas as pd
import argparse


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

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    # print(text_norm)
    return text_norm

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

def main(config):
    # 讀取sentence跟情緒等資訊
    data = pd.read_csv(config.input_csv_path, encoding='UTF-8')

    # 建立output路徑
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)

    # 製作帶情緒wav檔案至output路徑
    # emo = 0
    # for i in range(30):
    #     emo += 100
    #     msg, sr, audio = tts1('昔、かわいい小さな女の子がいました', emo)
    #     # msg, sr, audio = tts1(row['Sentence Japanese'], emo)
    #     sf.write(config.output_folder + 'output_' + "{:03d}".format(emo) + '.wav', audio, sr)
    #     print(emo)

    # emotional_list = {
    #     'Happy':123,
    #     'Angry':123,
    #     'Surprise':123,
    #     'Sad':123,
    #     'Fear':123,
    # }
    emotional_list = [
        102, # 激動偏平靜
        434, # 平靜一
        111, # 激動
        2077, # 小聲
        3550, # 平靜二
    ]

    for i, row_data in enumerate(data.iterrows()):
        row = row_data[1]
        emo = [row['Happy'], row['Angry'], row['Surprise'], row['Sad'], row['Fear']]
        for j, emotional_rate in enumerate(emo):
            if float(emotional_rate) != 0.0:
                emo = emotional_list[j]
                # print(emotional_rate)
                break
            else:
                emo = 3554

        msg, sr, audio = tts1(row['Sentence Japanese'], emo)
        sf.write(config.output_folder + 'output_' + "{:03d}".format(i) + '.wav', audio, sr)
        print(emo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_path', type=str, default='WordWithEmo.csv', help='')
    parser.add_argument('--output_folder', type=str, default='./OutputAudio/sample/', help='')
    config = parser.parse_args()
    
    main(config)
