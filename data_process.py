# coding: utf-8

import os
import collections

import pickle

import numpy as np
import yaml
import jieba

data_path = './data'
BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\'，。？；（）、！· '
UNK = 'unk'

limit = {
    'maxq': 20,
    'minq': 0,
    'maxa': 20,
    'mina': 3
}


def merge_data():
    """
    合并源数据
    :return:
    """
    file_list = os.listdir(os.path.join(data_path, 'raw_data'))
    conversations = []
    for file in file_list:
        if file == 'conversations.yml':
            continue
        with open(os.path.join(data_path, file), 'r') as f:
            content = yaml.load(f)
            conversations.extend(content['conversations'])
    for item in conversations:
        if len(item) != 2:
            print(item)
            conversations.remove(item)

    with open(os.path.join(data_path, 'corpus.yml'), 'w', encoding='utf8') as f:
        yaml.dump(conversations, f, allow_unicode=True)


def read_raw_corpus():
    """
    原始合并后的语料
    :return:
    """
    with open(os.path.join('corpus.yml'), encoding='utf-8') as f:
        conversations = yaml.load(f)
        return conversations


def filter_line(line):
    return ''.join([ch for ch in line if ch not in BLACKLIST]).strip()


def tokenize():
    """
    分词
    :return:
    """
    conversations = read_raw_corpus()

    np.random.shuffle(conversations)

    question_token = []
    answer_token = []
    conversations_token = []
    for item in conversations:
        question = filter_line(item[0])
        question = jieba.cut(question)
        answer = filter_line(item[1])
        answer = jieba.cut(answer)
        item = [' '.join(question), ' '.join(answer)]
        conversations_token.append(item)
        question_token.append(item[0].split(' '))
        answer_token.append(item[1].split(' '))

    with open(os.path.join('corpus_token.yml'), 'w') as f:
        yaml.dump(conversations_token, f, allow_unicode=True)

    return question_token, answer_token


def index_word(token_words):
    """
    建立映射
    :param token_words:
    :return:
    """
    words = sum(token_words, [])
    words_counter = collections.Counter(words)
    words_f = words_counter.most_common()
    freq_dist = dict(words_f)
    vocab_size = len(freq_dist)  # 由于数据比较少，把所有词汇都用上
    print('vocab_size {}'.format(vocab_size))
    index2word = ['_'] + [UNK] + [x[0] for x in words_f]
    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    print('index2word {}'.format(index2word[:10]))

    metadata = {
        'w2idx': word2index,
        'idx2w': index2word,
        'limit': limit,
        'freq_dist': freq_dist
    }

    # write to disk : data control dictionaries
    with open(os.path.join('metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    return index2word, word2index, freq_dist


def encode_word(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i][:limit['maxq']], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i][:limit['maxa']], w2idx, limit['maxa'])

        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    # 保存
    np.save(os.path.join('idx_q.npy'), idx_q)
    np.save(os.path.join('idx_a.npy'), idx_a)

    return idx_q, idx_a


def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0] * (maxlen - len(seq))


def pre_data():
    """
    预处理，结果存入pre_data
    :return:
    """
    # 1. 合并源数据
    # merge_data()
    # 2. 分词
    question_token, answer_token = tokenize()
    # 3. 建立索引映射
    index2word, word2index, freq_dist = index_word(question_token + answer_token)
    # 4. 编码问答句子对
    idx_q, idx_a = encode_word(question_token, answer_token, word2index)


###

def load_metadata():
    """
    获取元数据
    :return:
    """
    # path = os.path.join(data_path, 'pre_data', 'metadata.pkl')
    path = 'metadata.pkl'
    with open(path, 'rb') as f:
        metadata = pickle.load(f)

    return metadata


def load_qa_data():
    """
    获取问答数据
    :return:
    """
    # path = os.path.join(data_path, 'pre_data', 'idx_q.npy')
    path = 'idx_q.npy'
    idx_q = np.load(path)
    # path = os.path.join(data_path, 'pre_data', 'idx_a.npy')
    path = 'idx_a.npy'
    idx_a = np.load(path)

    return idx_q, idx_a


def split_dataset(x, y, ratio=[0.85, 0.15]):
    # number of examples
    data_len = len(x)
    lens = [int(data_len * item) for item in ratio]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0] + lens[1]], y[lens[0]:lens[0] + lens[1]]

    return (trainX, trainY), (testX, testY)


def decode(sequence, lookup, separator=''):
    """
    解码为文字
    :param sequence:
    :param lookup:
    :param separator:
    :return:
    """
    return separator.join([lookup[element] for element in sequence if element])


def decode_to_text(input, output, metadata, y=None):
    """
    解码为文字
    :return:
    """
    replies = []
    if y is not None:
        for ii, oi, yi in zip(input, output, y):
            q = decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
            a = decode(sequence=yi, lookup=metadata['idx2w'], separator=' ')
            decoded = decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')

            if decoded.count('unk') == 0:
                if decoded not in replies:
                    print('q : [{0}]; a : [{1}], y: [{2}]'.format(q, ' '.join(decoded), a))
                    replies.append(decoded)
    else:
        for ii, oi in zip(input, output):
            q = decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
            decoded = decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
            if decoded.count('unk') == 0:
                if decoded not in replies:
                    print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
                    replies.append(decoded)

    return replies


def upload_to_google_drive():
    """
    上传ckp到google drive
    :return: 
    """
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials

    # 1. Authenticate and create the PyDrive client.
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    import os
    dirlist = os.listdir('./')
    nums = []
    for f in dirlist:
        if '.meta' in f:
            num = f.split('.')[1].split('-')[1]
            num = int(num)
            print(num)
            nums.append(num)
    n_max = max(nums)
    print('max num {}'.format(n_max))
    f_meta = 'chatbot_model.ckpt-{}.meta'.format(n_max)
    f_data = 'chatbot_model.ckpt-{}.data-00000-of-00001'.format(n_max)
    f_index = 'chatbot_model.ckpt-{}.index'.format(n_max)
    print(f_meta, f_data, f_index)

    from googleapiclient.http import MediaFileUpload
    from googleapiclient.discovery import build
    drive_service = build('drive', 'v3')
    for f in [f_meta, f_data, f_index, 'checkpoint']:
        file_metadata = {
            'name': f,
            'mimeType': 'text/plain'
        }
        media = MediaFileUpload(f,
                                mimetype='text/plain',
                                resumable=True)
        created = drive_service.files().create(body=file_metadata,
                                               media_body=media,
                                               fields='id').execute()
        print('File ID: {}'.format(created.get('id')))


if __name__ == '__main__':
    pre_data()
    pass
    idx_q, idx_a = load_qa_data()
    metadata = load_metadata()

    decode_to_text(idx_q, idx_a, metadata)
