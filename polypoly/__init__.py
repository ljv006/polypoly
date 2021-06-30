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
        vocab_size = 0
        index2Word = []
        self.max_seq_len = 300
        with open(get_module_datapath('data/vocab.txt'), 'r', encoding='utf8') as vocab_file:
            for line in vocab_file:
                index2Word.append(line.strip())
                vocab_size += 1
        tag_size = 0
        with open(get_module_datapath('data/tag.txt'), 'r', encoding='utf8') as tag_file:
            for line in tag_file:
                tag_size += 1

        build_dict()#加载字典
        Model_path = get_module_datapath('savedModel/')
        self.model = tf.saved_model.load(Model_path)

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

    def process_input(self, inputs, max_len, batch_size = 1):
        with open(get_module_datapath('data/vocab.json'), 'r', encoding='utf-8') as vocab_f:
            wordIndexDict = json.load(vocab_f)

        def parse_fn(line):
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
            return (words, nwords, src_words), tags

        def generator_fn(inputs):
            for input_str in inputs:
                yield parse_fn(input_str)
        # Extract lines from input files using the Dataset API, can pass one filename or filename list
        shapes = ([max_len], (), [None]), [max_len]
        types = (tf.int64, tf.int64, tf.string), tf.int64
        dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn, inputs),output_shapes=shapes, output_types=types)
        dataset = dataset.batch(batch_size)
        return dataset

    def predict(self, str_list):
        input_texts = []
        phones = []
        # test_set = self.process_input(text_list, max_len = self.max_seq_len, batch_size = 32)
        test_set = self.process_input(inputs = str_list, max_len = self.max_seq_len)
        for (test_input_data, test_nwords, test_src_words), test_tags in test_set:
            # pred_tags_list = self.session.run([self.viterbi_sequence],feed_dict={self.input_data: test_input_data, self.tags:test_tags, self.original_sequence_lengths: test_nwords})
            # features = {"input_data":np.array([test_input_data]), "tags":np.array([test_tags]), "original_sequence_lengths":np.array(test_nwords)}

            pred_tags_list = self.model.signatures['serving_default'](original_sequence_lengths = test_nwords, input_data = test_input_data, tags = test_tags)['viterbi_sequence']
            
            pred_tags = np.argmax(pred_tags_list[0], 1)
            src_words = []
            
            for token in test_src_words[0]:
                # pdb.set_trace()
                word = token.numpy().decode('utf8')
                if word == '<pad>':
                    continue
                else:
                    src_words.append(word)
            text = ''.join(src_words)
            model_preds = []
            for label in pred_tags:
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

