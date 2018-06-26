import pparser
import os
import sys
import numpy as np


class DataUtil:
    _PAD_ = "_PAD_"  # 빈칸 채우는 심볼
    _STA_ = "_STA_"  # 디코드 입력 시퀀스의 시작 심볼
    _EOS_ = "_EOS_"  # 디코드 입출력 시퀀스의 종료 심볼
    _UNK_ = "_UNK_"  # 사전에 없는 단어를 나타내는 심볼

    _PAD_ID_ = 0
    _STA_ID_ = 1
    _EOS_ID_ = 2
    _UNK_ID_ = 3
    _PRE_DEFINED_ = [_PAD_ID_, _STA_ID_, _EOS_ID_, _UNK_ID_]
    _UNK_LIST = ['unk', 'unk1', 'unk2', 'unk3', 'unk4', 'unk5', 'unk6', 'unk7', 'unk8']

    def __init__(self):
        self.parser = pparser.ret_parser()
        self.vocab_dict = {}
        self.examples = []
        self.ans = []
        self.vocab_list = self._PRE_DEFINED_ + []
        self._index_in_epoch = 0
        # 사전 불러오기
        with open('./word/word.txt', 'r', encoding='utf-8-sig') as vocab_file:
            for line in vocab_file:
                self.vocab_list.append(line.strip())
        self.vocab_dict = {n: i for i, n in enumerate(self.vocab_list)}
        self.vocab_size = len(self.vocab_list)

    def cut_eos(self, indices):
        eos_idx = indices.index(self._EOS_ID_)
        return indices[:eos_idx]

    def is_eos(self, voc_id):
        return voc_id == self._EOS_ID_

    def is_defined(self, voc_id):
        return voc_id in self._PRE_DEFINED_

    def max_len(self, batch_set):
        max_len_input = 0
        max_len_output = 0

        for i in range(0, len(batch_set) - 1, 2):
            len_input = len(batch_set[i])
            len_output = len(batch_set[i + 1])
            if len_input > max_len_input:
                max_len_input = len_input
            if len_output > max_len_output:
                max_len_output = len_output

        return max_len_input, max_len_output + 1

    def pad(self, seq, max_len, start=None, eos=None):
        if start is True:
            padded_seq = [self._STA_ID_] + seq
        elif eos is True:
            padded_seq = seq + [self._EOS_ID_]
        else:
            padded_seq = seq

        if len(padded_seq) < max_len:
            return padded_seq + ([self._PAD_ID_] * (max_len - len(padded_seq)))
        else:
            return padded_seq

    def pad_left(self, seq, max_len):
        if len(seq) < max_len:
            return ([self._PAD_ID_] * (max_len - len(seq))) + seq
        else:
            return seq

    def transform(self, input, input_max):
        # int id를 one hot vector로 바꾸는 함수
        enc_input = self.pad(input, input_max)

        enc_input.reverse()

        enc_input = np.eye(self.vocab_size)[enc_input]

        return enc_input

    def next_batch(self, batch_size):
        enc_input = []
        target = np.ndarray(shape=[batch_size, 2], dtype=np.float32)

        start = self._index_in_epoch

        if self._index_in_epoch + batch_size < len(self.examples) - 1:
            self._index_in_epoch = self._index_in_epoch + batch_size
        else:
            self._index_in_epoch = 0

        batch_set = self.examples[start:start + batch_size]
        batch_ans = self.ans[start:start + batch_size]

        max_len_input, max_len_output = self.max_len(batch_set)
        if max_len_input < max_len_output:
            max_len_input = max_len_output
        for i in range(0, len(batch_set)):
            enc = self.transform(batch_set[i], max_len_input)
            enc_input.append(enc)
            target[i][batch_ans[i]] = 1
        return enc_input, target

    def replace_unknown(self, word, type='enc', loc=0):
        # _UNK_를 들어온 단어로 복원하기 위한 함수
        # type은 어느 언노운 사전에 저장할지, loc는 dec 사전에서만 쓰이는 것으로 어디에 저장할지를 정한다.
        if type == 'enc':
            self.enc_unknowndic[self._UNK_LIST[self.dicsize]] = word
            self.dicsize += 1
            if self.dicsize == len(Dialog._UNK_LIST):
                self.dicsize -= 1
            return self._UNK_LIST[self.dicsize - 1]
        elif type == 'dec':
            if word in self._UNK_LIST:
                word = self.enc_unknowndic.get(word, word)
            self.dec_unknowndic[self._UNK_LIST[loc]] = word
            return word
        else:
            raise ValueError('something is wrong')

    def unknown_dic_clear(self):
        # unknown dic을 비워준다.
        self.enc_unknowndic.clear()
        self.dec_unknowndic.clear()
        self.dicsize = 0

    def tokens_to_ids(self, tokens):
        ids = []
        for t in tokens:
            if t in self.vocab_dict:
                ids.append(self.vocab_dict[t])
            else:
                w = self.replace_unknown(t)
                if self.debug is True:
                    print('new enc unk : ', t, '->', w)
                ids.append(self.vocab_dict[w])
        return ids

    def ids_print(self, ids):
        for i in ids:
            if i == self._UNK_ID_:
                print("unk", end=' ')
            else:
                print(str(i), end=' ')
        print()

    def ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.vocab_list[i])
        return tokens

    def tokenizer(self, code, findWord="", keep=False):
        # 코드를 입력받아 토큰화
        ast = self.parser.python_to_tree(code)
        ret = self.parser.tree_to_list(ast)
        if findWord in ret:
            print(ast.pretty())
        # val을 비워준다.
        if keep is False:
            self.parser.clear_dict()
        return ret

    def load_example(self):
        self.examples = []
        self.ans = []
        for (path, dir, files) in os.walk('./data/'):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.py':
                    ast = self.parser.python_to_tree("%s/%s" % (path, filename))
                    tlist = self.parser.tree_to_list(ast)
                    ids = self.tokens_to_ids(tlist)
                    self.examples.append(ids)
                    self.ans.append(int(path[7]))
                    self.parser.clear_dict()


tot = 0

# 이것을 직접 실행하면 data 밑에 있는 모든 코드를 읽어서 word 화 해준다.
if __name__ == '__main__':
    tarword = None
    try:
        tarword = sys.argv[1]
    except:
        print('검색어 없음')
    p = pparser.ret_parser()
    tword = []
    for (path, dir, files) in os.walk('./data/'):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.py':
                ast = p.python_to_tree("%s/%s" % (path, filename))
                print(filename)
                tlist = p.tree_to_list(ast)
                if tarword in tlist:
                    print("%s/%s" % (path, filename))
                tword += tlist
                p.clear_dict()
                tot += 1
    print(tot)
    words = sorted(list(set(tword)))
    with open('./word/word.txt', 'w') as vocab_file:
        for w in words:
            vocab_file.write(w + '\n')
