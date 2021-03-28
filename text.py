from segmentation import seg
import tqdm


def get_all_vocab(data_frame):
    vocab = []
    for item in tqdm.tqdm(data_frame.iterrows()):
        vocab = vocab + [x for x in seg(item[1].content)]
    return vocab


def generate_vector(vocab, words):
    return [words.count(x) for x in vocab]


if __name__ == '__main__':
    from data import load_data
    print(generate_vector(["你","我","他"], ["你","我","你"]))
