import os
from config import data_dir
import pandas as pd
import random
import tqdm

classes = os.listdir(data_dir)


def get_files(directory, number):
    path_dir = os.listdir(directory)  # 取图片的原始路径
    pick_number = int(number)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(path_dir, pick_number)  # 随机选取picknumber数量的样本图片
    return sample


def load_data(number_each_class=100):
    data_info = pd.DataFrame(columns=["content", "class"])
    for cla in tqdm.tqdm(classes):
        for file in get_files(data_dir + "/" + cla, number_each_class):
            with open(os.path.join(data_dir, cla, file), 'r') as f:
                data_info = data_info.append({"content": f.read(), "class": cla}, ignore_index=True)
    return data_info

