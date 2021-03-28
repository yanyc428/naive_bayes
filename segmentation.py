import jieba
from config import stop_words_path

with open(stop_words_path, 'r') as f:
    stop_words = f.read().split()


def seg(content):
    tmp = jieba.cut(content)
    return [x for x in tmp if x not in stop_words]

