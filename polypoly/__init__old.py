#encoding=utf8
import functools
import json
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pdb
import random
import sys
import time
import warnings

import jieba
import numpy as np
import pkg_resources
import tensorflow as tf
from pypinyin import Style, lazy_pinyin, pinyin
from tensorflow.saved_model import tag_constants

warnings.filterwarnings('ignore')

get_module_datapath = lambda *res: pkg_resources.resource_filename(__name__, os.path.join(*res))

def rule_based_func(input_str, phone):
    # input_str = '谁知道郑屠拿到不但不应'
    word_phone_map = {'应':'ying4','着':'zhe5','咯':'lo5','少':'shao3','蜇':'zhe1','熨':'yun4','掖':'ye1','幢':'zhuang4','耶':'ye2','忪':'zhong1','蹊':'xi1','塞':'sai1','处':'chu3','哟':'yo1','搂':'lou3','椎':'chui2', '枞':'cong1','茄':'qie2','偈':'ji4','桧':'gui4','鹄':'hu2','喷':'pen1','秘':'mi4','孱':'chan2','逮':'dai3','提':'ti2','偻':'lv3','缪':'miao4','蔓':'man4','磅':'bang4','膀':'pang1','扛':'kang2','卜':'bu3','燎':'liao3','咳':'hai1','晕':'yun1','喽':'lou5','予':'yu3','颤':'chan4','济':'ji4','系':'ji4','参':'can1','囤':'tun2','混':'hun4','熬':'ao2','裳':'chang2','结':'jie2','担':'dan1','觉':'jiao4','脯':'fu3','剥':'bao1','桔':'ju2','攒':'zan3','咋':'za3','绿':'lv4','烙':'lao4','伯':'bo2', '吁':'xu1','待':'dai1','坊':'fang1','呢':'ne5', '泡':'pao4','咧':'lie5','贾':'jia3'}
    words = jieba.lcut(input_str, cut_all=False)
    final_phone = []
    index = 0
    #针对单字进行调整
    for w in words:
        if len(w) == 1:#应对单字
            if w in word_phone_map:
                final_phone.append(word_phone_map[w])
            else:
                final_phone.extend(phone[index: index + len(w)])
        else:
            final_phone.extend(phone[index: index + len(w)])
        index += len(w)
    #针对变调进行调整
    words = list(input_str)
    for index, w in enumerate(words):
        if w == '一':
            if index - 1 >= 0 and ('第' == words[index - 1] or '初' == words[index - 1] or '十' == words[index - 1] or '周' == words[index - 1]):
                final_phone[index] ='yi1'
            elif index - 2 >= 0 and ('星' == words[index - 2] and '期' == words[index - 1]) or ('礼' == words[index - 2] and '拜' == words[index - 1]):
                final_phone[index] ='yi1'
            elif index + 1 <= len(words) - 1 and '4' in final_phone[index + 1]:
                final_phone[index] = 'yi2'
            elif index + 1 <= len(words) - 1 and ('1' in final_phone[index + 1] or '2' in final_phone[index + 1] or '3' in final_phone[index + 1]):
                final_phone[index] = 'yi4'
            else:
                final_phone[index] ='yi1'
        if w == '不' and final_phone[index] == 'bu4':
            if index + 1 <= len(words) - 1 and '4' in final_phone[index + 1]:
                final_phone[index] = 'bu2'
    # pdb.set_trace()
    return final_phone

def build_dict():
    words = []
    words_map = {}
    with open(get_module_datapath('data/proun_dict_tiny.txt'),'r', encoding='utf8') as dict_file, open(get_module_datapath('data/dict.json'), 'w', encoding='utf8') as output_file, open(get_module_datapath('data/words.txt'), 'w', encoding = 'utf8') as word_file:
        for line in dict_file:
            tmp_list = line.split()
            word = tmp_list[0]
            words.append(word)
            if word not in words_map:
                phone = ''
                # pdb.set_trace()
                try:
                    for i in range(len(word)):
                        phone += tmp_list[i + 1]  + ' '
                except:
                    continue
                phone = phone.strip()
                words_map[word] = phone
        for w in words:
            if len(w) >= 2:
                word_file.writelines(w + '\n')
        json.dump(words_map, output_file)

def lookup_dict(word, dict_file):
    if word in dict_file:
        return dict_file[word].split()
    else:
        return lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)

class Polypoly(object):
    def __init__(self):
        model_params = {
        "learning_rate": 0.003,
        "input_size": 1,
        "embedding_size": 128,
        "num_units": 256,
        "dropout": 0.1,
        "train_batch_size": 512,
        "valid_batch_size": 128,
        "training_epoch": 40,
        "max_seq_len": 300
        }
        vocab_size = 0
        index2Word = []
        with open(get_module_datapath('data/vocab.txt'), 'r', encoding='utf8') as vocab_file:
            for line in vocab_file:
                index2Word.append(line.strip())
                vocab_size += 1
        tag_size = 0
        with open(get_module_datapath('data/tag.txt'), 'r', encoding='utf8') as tag_file:
            for line in tag_file:
                tag_size += 1

        build_dict()#加载字典
        MODEL_PATH = get_module_datapath('savedModel/')
        self.max_seq_len = model_params['max_seq_len']
        num_units = model_params['num_units']
        embedding_size = model_params['embedding_size']
        number_of_classes = tag_size
        self.input_data = tf.placeholder(tf.int64, [None, self.max_seq_len], name="input_data") # shape = (batch, batch_seq_len, input_size)
        embedding = tf.get_variable('embedding', [vocab_size, embedding_size])
        inputs_embedded = tf.nn.embedding_lookup(embedding, self.input_data)
        self.tags = tf.placeholder(tf.int64, shape=[None, self.max_seq_len], name="tags") # shape = (batch, sentence)

        self.original_sequence_lengths = tf.placeholder(tf.int64, [None])

        with tf.name_scope("BiLSTM"):
            with tf.variable_scope('forward'):
                lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
            with tf.variable_scope('backward'):
                lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
            (output_fw, output_bw), ((_, output_fw_h), (_, output_bw_h)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, 
                                                                            cell_bw=lstm_bw_cell, 
                                                                            inputs=inputs_embedded,
                                                                            sequence_length=self.original_sequence_lengths, 
                                                                            dtype=tf.float32,
                                                                            scope="BiLSTM")

        # As we have a Bi-LSTM, we have two outputs which are not connected, so we need to merge them.
        outputs = tf.concat([output_fw, output_bw], axis=2)
        outputs_flat = tf.reshape(outputs, [-1, 2 * num_units])

        pred = tf.compat.v1.layers.dense(outputs_flat, 64, activation='relu')
        pred = tf.compat.v1.layers.dense(pred, 16, activation='relu')
        pred = tf.compat.v1.layers.dense(pred, number_of_classes, activation=None)

        scores = tf.reshape(pred, [-1, self.max_seq_len, number_of_classes])

        self.viterbi_sequence = tf.nn.softmax(scores, 2)
        self.session = tf.Session()
        # saver = tf.train.Saver()
        # saver.restore(self.session, MODEL_PATH)
        tf.saved_model.loader.load(
            self.session,
            [tag_constants.SERVING],
            MODEL_PATH,
        )
        jieba.load_userdict(get_module_datapath('data/words.txt'))
        with open(get_module_datapath('data/dict.json'), 'r', encoding='utf-8') as dict_f:
            self.wordPhoneMap = json.load(dict_f)
        self.high_freq_poly_words = []
        with open(get_module_datapath('data/high_freq_poly_words.txt'), 'r', encoding='utf-8') as poly_words:
            for line in poly_words:
                self.high_freq_poly_words.append(line.strip())
        self.tagIndexMap = []
        with open(get_module_datapath('data/tag.txt'), 'r', encoding = 'utf8') as tag_file:
            for line in tag_file:
                self.tagIndexMap.append(line.strip())

    def process_input(self, text_list, max_len, batch_size = 32):
        if os.path.exists(get_module_datapath('data/vocab.json')):
            with open(get_module_datapath('data/vocab.json'), 'r', encoding='utf-8') as vocab_f:
                wordIndexDict = json.load(vocab_f)
        result_list = []
        word_list = []
        src_word_list = []
        tag_list = []
        nword_list = []
        for line in text_list:
            text = line.strip()
            words = [wordIndexDict[w] if w in wordIndexDict else wordIndexDict['<unk>'] for w in list(text)]
            nwords = len(words)
            tags = [1 for t in range(nwords)]
            src_words = [w.encode() for w in list(text)]
            ntags = len(tags)
            if nwords < max_len:
                words.extend([0 for i in range(max_len - nwords)])
                src_words.extend('<pad>'.encode() for i in range(max_len - nwords))
                tags.extend([0 for i in range(max_len - ntags)])
            else:
                words = words[:max_len]
                src_words = src_words[:max_len]
                tags = tags[:max_len]
            # tmp_list.append(((words, nwords, src_words), tags))
            word_list.append(words)
            src_word_list.append(src_words)
            nword_list.append(nwords)
            tag_list.append(tags)
            # pdb.set_trace()
            if len(word_list) == batch_size:
                result_list.append(((word_list, nword_list, src_word_list),tag_list))
                word_list = []
                src_word_list = []
                tag_list = []
                nword_list = []
        return np.array(result_list) if result_list else np.array([((word_list, nword_list, src_word_list),tag_list)])

    def predict(self, text_list):
        input_texts = []
        phones = []
        test_set = self.process_input(text_list, max_len = self.max_seq_len, batch_size = 32)
        preds = []
        for (test_input_data, test_nwords, test_src_words), test_tags in test_set:
            pred_tags_list = self.session.run([self.viterbi_sequence],feed_dict={self.input_data: test_input_data, self.tags:test_tags, self.original_sequence_lengths: test_nwords})
            pred_tags = pred_tags_list[0]
            # pdb.set_trace()
            for i in range(len(pred_tags)):
                preds.append({'src_words':test_src_words[i], 'tags':pred_tags[i]})
        # with open(result_file_path, 'w', encoding='utf8') as result_file:
            # pdb.set_trace()
        for item in preds:
            text = []
            for w in item['src_words']:
                # word = index2Word[w]
                word = w.decode('utf8')
                if word == '<pad>':
                    continue
                else:
                    text.append(word)
            text = ''.join(text)
            model_pred_labels = np.argmax(item['tags'],1)
            model_preds = []
            # pdb.set_trace()
            for label in model_pred_labels:
                if label == 0:
                    continue
                model_preds.append(self.tagIndexMap[label])

            words = jieba.lcut(text, cut_all=False)
            phone = []
            cur_idx = 0
            # pdb.set_trace()
            for w_idx, word in enumerate(words):
                if word in self.wordPhoneMap and word not in self.high_freq_poly_words:#直接命中字典但不是高频多音字，直接使用字典的结果
                    phone.extend(self.wordPhoneMap[word].split())
                elif word in self.wordPhoneMap and word in self.high_freq_poly_words:#直接命中字典但是高频多音字，使用模型的结果
                    try:
                        model_tag = model_preds[cur_idx]
                    except:
                        pdb.set_trace()
                    if word in model_tag:
                        phone.append(model_tag.replace(word + '|', ''))
                    else:
                        # phone.append(lazy_pinyin(w, style=Style.TONE3, neutral_tone_with_five=True)[0])
                        phone.extend(lookup_dict(word, self.wordPhoneMap))
                elif word not in self.wordPhoneMap:#既不在字典中也不是高频多音字，进一步拆分
                    for token_idx, token in enumerate(word):
                        if token in self.high_freq_poly_words:#拆分后是高频多音字，使用模型结果
                            try:
                                model_tag = model_preds[cur_idx + token_idx]
                            except:
                                pdb.set_trace()
                            if token in model_tag:
                                phone.append(model_tag.replace(token + '|', ''))
                            else:
                                phone.extend(lookup_dict(token, self.wordPhoneMap))
                        else:#拆分后不是高频多音字，使用字典的结果
                            # phone.append(lazy_pinyin(w, style=Style.TONE3, neutral_tone_with_five=True)[0])
                            phone.extend(lookup_dict(token, self.wordPhoneMap))
                cur_idx += len(word)

            #使用规则
            final_phone = rule_based_func(text, phone)
            # pdb.set_trace()
            input_texts.append(text)
            phones.append(final_phone) 
        return input_texts, phones

polypoly = Polypoly()

predict = polypoly.predict

