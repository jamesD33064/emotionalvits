import gradio as gr
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import numpy as np


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm
hps = utils.get_hparams_from_file("./configs/vtubers.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
_ = net_g.eval()

_ = utils.load_checkpoint("./G_49200.pth", net_g, None)
all_emotions = np.load("all_emotions.npy")
emotion_dict = {
    "小声": 2077,
    "激动": 111,
    "平静1": 434,
    "平静2": 3554
}
import random
def tts(txt, emotion):
    stn_tst = get_text(txt, hps)
    randsample = None
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([0])
        if type(emotion) ==int:
            emo = torch.FloatTensor(all_emotions[emotion]).unsqueeze(0)
        elif emotion == "random":
            emo = torch.randn([1,1024])
        elif emotion == "random_sample":
            randint = random.randint(0, all_emotions.shape[0])
            emo = torch.FloatTensor(all_emotions[randint]).unsqueeze(0)
            randsample = randint
        elif emotion.endswith("wav"):
            import emotion_extract
            emo = torch.FloatTensor(emotion_extract.extract_wav(emotion))
        else:
            emo = torch.FloatTensor(all_emotions[emotion_dict[emotion]]).unsqueeze(0)

        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1, emo=emo)[0][0,0].data.float().numpy()
    return audio, randsample


def tts1(text, emotion):
    if len(text) > 150:
        return "Error: Text is too long", None
    audio, _ = tts(text, emotion)
    return "Success", (hps.data.sampling_rate, audio)

def tts2(text):
    if len(text) > 150:
        return "Error: Text is too long", None
    audio, randsample = tts(text, "random_sample")

    return str(randsample), (hps.data.sampling_rate, audio)

def tts3(text, sample):
    if len(text) > 150:
        return "Error: Text is too long", None
    try:
        audio, _ = tts(text, int(sample))
        return "Success", (hps.data.sampling_rate, audio)
    except:
        return "输入参数不为整数或其他错误", None
app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("使用预制情感合成"):
            tts_input1 = gr.TextArea(label="日语文本", value="こんにちは。私わあやちねねです。")
            tts_input2 = gr.Dropdown(label="情感", choices=list(emotion_dict.keys()),  value="平静1")
            tts_submit = gr.Button("合成音频", variant="primary")
            tts_output1 = gr.Textbox(label="Message")
            tts_output2 = gr.Audio(label="Output")
            tts_submit.click(tts1, [tts_input1, tts_input2], [tts_output1, tts_output2])
        with gr.TabItem("随机抽取训练集样本作为情感参数"):
            tts_input1 = gr.TextArea(label="日语文本", value="こんにちは。私わあやちねねです。")
            tts_submit = gr.Button("合成音频", variant="primary")
            tts_output1 = gr.Textbox(label="随机样本id（可用于第三个tab中合成）")
            tts_output2 = gr.Audio(label="Output")
            tts_submit.click(tts2, [tts_input1], [tts_output1, tts_output2])

        with gr.TabItem("使用情感样本id作为情感参数"):

            tts_input1 = gr.TextArea(label="日语文本", value="こんにちは。私わあやちねねです。")
            tts_input2 = gr.Number(label="情感样本id", value=2004)
            tts_submit = gr.Button("合成音频", variant="primary")
            tts_output1 = gr.Textbox(label="Message")
            tts_output2 = gr.Audio(label="Output")
            tts_submit.click(tts3, [tts_input1, tts_input2], [tts_output1, tts_output2])

        with gr.TabItem("使用参考音频作为情感参数"):
            tts_input1 = gr.TextArea(label="text", value="暂未实现")

    app.launch()
