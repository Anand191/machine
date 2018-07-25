import os
import argparse
import logging
import numpy as np
import random
import torch
import torchtext
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seq2seq
from seq2seq.evaluator.predictor2 import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField, AttentionField

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
parser.add_argument('--test', help='Path to test data')
parser.add_argument('--output_dir', help='Path to save results')
parser.add_argument('--run_infer', action='store_true')
parser.add_argument('--max_plots', type=int, help='Maximum sequence length', default=10)
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--attention', choices=['pre-rnn', 'post-rnn'], default=False)
parser.add_argument('--attention_method', choices=['dot', 'mlp', 'hard'], default=None)
parser.add_argument('--use_attention_loss', action='store_true')
parser.add_argument('--teacher_forcing_ratio', type=float, help='Teacher forcing ratio', default=0.2)
parser.add_argument('--ignore_output_eos', action='store_true', help='Ignore end of sequence token during training and evaluation')

random.seed(100)
opt = parser.parse_args()
test_splits = ['unseen', 'longer', 'unseen_longer']
if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

IGNORE_INDEX=-1
use_output_eos = not opt.ignore_output_eos
src = SourceField()
tgt = TargetField(use_output_eos)
attn = AttentionField(use_vocab=False, ignore_index=IGNORE_INDEX)
max_len = opt.max_len


def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

def getarr(data):
    data_src = []
    data_tgt = []
    data_attn = []
    for i in range(len(data)):
        data_src.append(vars(data[i])[seq2seq.src_field_name])
        data_tgt.append(vars(data[i])[seq2seq.tgt_field_name])
        if opt.use_attention_loss or opt.attention_method == 'hard':
            data_attn.append(vars(data[i])[seq2seq.attn_field_name])
    if(len(data_attn) !=0):
        master_data = np.zeros((len(data_src),3), dtype=object)
    else:
        master_data = np.zeros((len(data_src), 2), dtype=object)
    for i in range(len(data_src)):
        master_data[i, 0] = ' '.join(map(str, data_src[i]))
        master_data[i, 1] = ' '.join(map(str, data_tgt[i]))
        if (len(data_attn) != 0):
            master_data[i, 2] = data_attn[i]
    return master_data

def load_model(checkpoint_path):
    logging.info("loading checkpoint from {}".format(os.path.join(checkpoint_path)))
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    src.vocab = input_vocab
    output_vocab = checkpoint.output_vocab
    tgt.vocab = output_vocab
    tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
    tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]
    return seq2seq, input_vocab, output_vocab

def prepare_data(data_path):
    # generate training and testing data
    tabular_data_fields = [('src', src), ('tgt', tgt)]
    if opt.use_attention_loss or opt.attention_method == 'hard':
        tabular_data_fields.append(('attn', attn))
    gen_data = torchtext.data.TabularDataset(
        path=data_path, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    )
    data_arr = getarr(gen_data)

    return data_arr

def get_colour(input_seq, target_seq, prediction, real_pred):
    colour = []
    if ('verify' in input_seq):
        for j in real_pred:
            if (j in target_seq):
                colour.append('g')
            else:
                colour.append('r')
    elif ('produce' in input_seq):
        if ('or' in input_seq):
            any_in = lambda a, b: any(i in b for i in a)
            if any_in(input_seq, prediction):
                colour.extend(['g'] * len(real_pred))
            else:
                colour.extend(['r'] * len(real_pred))
        elif('not' in input_seq):
            all_nots = []
            all_ands = []
            if(input_seq[0] != 'not'):
                all_ands.append(input_seq[0])
            for i in range(1, len(input_seq)-1):
                if(input_seq[i-1]=='not'):
                    all_nots.append(input_seq)
                elif(input_seq[i-1]=='and' and input_seq[i] != 'not'):
                    all_ands.append(input_seq[i])
            for i in range(len(prediction)-1):
                if(prediction[i] in all_ands):
                    colour.append('g')
                if (prediction[i] in all_nots):
                    colour.append('r')
                else:
                    colour.append('g')
            if(prediction[-1] == '<eos>'):
                colour.append('g')
        else:
            for p in prediction:
                if(p in input_seq):
                    colour.append('g')
                else:
                    colour.append('r')
    return colour

def showAttention(input_sentence, output_words, attentions,name,colour):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(10,14))
    ax = fig.add_subplot(111)
    ax.yaxis.tick_right()
    cax = ax.matshow(attentions, cmap='bone', vmin=0, vmax=1)
    #fig.colorbar(cax)
    cbaxes = fig.add_axes([0.05, 0.1, 0.03, 0.8])
    cb = plt.colorbar(cax, cax=cbaxes)
    cbaxes.yaxis.set_ticks_position('left')

    # Set up axes
    ax.set_xticks(np.arange(len(input_sentence.split())+1))
    ax.set_yticks(np.arange(len(output_words)+1))
    ax.set_xticklabels([''] + input_sentence.split(' '), rotation=45) #+['<EOS>']
    ax.set_yticklabels([''] + output_words)
    # print(ax.get_yticklabels())
    # input()
    #Colour ticks
    for ytick, color in zip(ax.get_yticklabels()[1:], colour):
        ytick.set_color(color)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #X and Y labels
    ax.set_xlabel("INPUT")
    ax.set_ylabel("OUTPUT")
    ax.yaxis.set_label_position('right')
    ax.xaxis.set_label_position('top')

    plt.savefig("{}-eps.eps".format(name), format='eps')
    plt.close(fig)
    # plt.show()
    # input()

def plot_attention(idxs, test_data,opt_path):
    print("Begin Plotting")
    for x in idxs:
        ipt_sentence = test_data[x, 0]
        opt_sentence = test_data[x,1]
        seq = ipt_sentence.strip().split()
        tgt_seq = opt_sentence.strip().split()
        ipt_seq = ipt_sentence.strip().split()
        if opt.attention_method == 'hard':
            tgt_var = list(map(int, test_data[x, 2]))
            outputs, attention = predictor.predict(seq, tgt_seq,tgt_var)
        else:
            outputs, attention = predictor.predict(seq)
        trimmed_prediction = [pcl for pcl in outputs if pcl != 'erm']
        colour = get_colour(ipt_seq, tgt_seq, trimmed_prediction, outputs)
        name = os.path.join(opt_path, 'plot' + '{}'.format(x))
        showAttention(ipt_sentence, outputs, attention, name, colour)


training_type = '_'.join(map(str,opt.checkpoint_path.split('/')[-3].split('_')[0:2]))
model, input_vocab, output_vocab = load_model(opt.checkpoint_path)
predictor = Predictor(model, input_vocab, output_vocab)
for sub in (test_splits):
    test_path = os.path.join(opt.test,'Verify_Produce_{}.tsv'.format(sub))
    test_data= prepare_data(test_path)
    idxs = random.sample(np.arange(test_data.shape[0]).tolist(), 20)
    #idxs = [3, 4, 6, 10, 16, 25, 28, 31, 37, 40]
    opt_path = os.path.join(opt.output_dir,training_type,sub)
    if not os.path.exists(opt_path):
        os.makedirs(opt_path)
    plot_attention(idxs, test_data,opt_path)

    print("finished plotting for mode={}".format(sub))