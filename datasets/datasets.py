#!usr/bin/python3
# -*- coding: utf-8 -*- 
"""
Project: resnet_zml
File: 01prepare_data_with_class_label.py
IDE: PyCharm
Creator: morei
Email: zhangmengleiba361@163.com
Create time: 2021-02-20 16:34
Introduction:
This script is to compress image and prepare the class_label csv file for data group
"""

import os
import pandas as pd
import random
import json
from torchvision import transforms as T
from pathlib import Path
from PIL import Image as pil_image
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff']


def open_image(image_path):
    """
    :param image_path:
    :return:
    """
    try:
        pil_image.open(image_path)
        return True
    except Exception as e:
        return None


def resize(image_path, short_len=512):
    img = pil_image.open(image_path)
    transform1 = T.Compose([
        T.Resize(short_len)
    ])
    r_img = transform1(img)
    return r_img


def df_to_json(df):
    map1 = {}
    for i in range(len(df)):
        map1[str(df.iat[i, 0])] = str(df.iat[i, 1])
    js = json.dumps(map1, indent=2, ensure_ascii=False)
    return js


class ImagePath:
    """
    加载数据
    """

    def __init__(self, ):
        """主要目标： 获取所有图片路径，并根据训练，验证，测试划分数据
        """
        self.extension = IMG_EXTENSIONS
        self.img_path = []
        self.img_fail = []
        self.dirs = []

    def get_images(self, root, new_or_old='原始'):
        """获取所有的图片
            :param new_or_old:
            :param root:
        """
        # 获取图片路径
        for rs, ds, fs in os.walk(root):
            for f in fs:
                img_path = os.path.join(rs, f)
                d = img_path.split(os.sep)[-2]
                if any(img_path.lower().endswith(ext) for ext in self.extension) and open_image(img_path):
                    self.img_path.append(img_path)
                    self.dirs.append(d)
                else:
                    self.img_fail.append(img_path)
        image_df = pd.DataFrame({"img_path": self.img_path,
                                 "最小子文件夹名称": self.dirs
                                 })
        out_f1 = './{}图片路径.csv'.format(new_or_old)
        print('{}已经生成完毕！'.format(out_f1))
        image_df.to_csv(out_f1, index=False, encoding="UTF-8-sig")
        if len(self.img_fail) > 0:
            out_f2 = './不合格的图片.csv'
            pd.DataFrame(self.img_fail).to_csv(out_f2, index=False, encoding="UTF-8-sig")
            print('{}已经生成完毕！'.format(out_f2))
        if new_or_old == '原始':
            # 统计图片张数
            dirs_count_df = image_df['最小子文件夹名称'].value_counts().reset_index()
            dirs_count_df.columns = ['最小子文件夹名称', '图片张数']
            # 按识农流程生成核心信息表
            dirs_count_df[['标签数字', '百科条目', '百科ID_来自boss系统', 'PyTorch模型输入输出标签']] = None
            out_f3 = './待添加百科信息的核心信息表.csv'
            print('{}已经生成完毕！'.format(out_f3))
            dirs_count_df.to_csv(out_f3, index=False, encoding="UTF-8-sig")

    @staticmethod
    def resize_to_folder(image_df, root, short_len=512):
        if root.endswith(os.sep):
            root = root[:len(root) - 1]
        image_df["washed_tf_path"] = image_df.img_path.apply(
            lambda x: os.path.join((root + str(short_len)), x[len(root) + 1:]))
        ori_file_lst = image_df['img_path'].values.tolist()
        des_file_lst = image_df['washed_tf_path'].values.tolist()
        for i in tqdm(range(len(image_df))):
            if not Path(str(Path(des_file_lst[i]).parent)).is_dir():
                os.makedirs(str(Path(des_file_lst[i]).parent))
            try:
                n_image = resize(ori_file_lst[i])
                n_image.save(str(des_file_lst[i]))  # 目标路径要与百科条目一致
            except Exception as r:
                print('未知错误 %s' % r)
        print('{}最短边{}图片已经生成完毕！'.format(root, short_len))

    @staticmethod
    def add_torch_id(csv_wiki):
        # 导入完整的核心信息表
        """
        核心信息表示例
        ['最小子文件夹名称', '图片张数', '标签数字', '百科条目', '百科ID_来自boss系统', 'PyTorch模型输入输出标签']
        """
        wiki_df = pd.read_csv(csv_wiki)
        wiki_df = wiki_df[~wiki_df['标签数字'].isin([-1])]
        l_id = wiki_df['标签数字'].values.tolist()
        l_id_u = sorted(set(l_id), key=l_id.index)
        torch_id = list(range(0, len(l_id_u)))
        l_t_dic = dict(zip(l_id_u, torch_id))
        t = l_t_dic[2]  # 标签数字和PyTorch对应
        wiki_df['PyTorch模型输入输出标签'] = wiki_df.标签数字.apply(lambda x: l_t_dic[x])
        # 按识农流程生成核心信息表
        out_f = './添加pytorchID的核心信息表.csv'
        wiki_df.to_csv(out_f, index=False, encoding="UTF-8-sig")
        print('{}已经生成完毕！'.format(out_f))
        # 生成wiki_pytorch.json
        # PyTorch模型输入输出标签_百科ID
        t_w_id = wiki_df[['PyTorch模型输入输出标签', '百科ID_来自boss系统']]
        t_w_id_json = df_to_json(t_w_id)
        with open('../output/torchID_wikiID.json', 'w') as f_json:
            f_json.write(t_w_id_json)
            print('{}已经生成完毕！'.format(f_json))
        # PyTorch模型输入输出标签_百科条目
        t_w = wiki_df[['PyTorch模型输入输出标签', '百科条目']]
        t_w_json = df_to_json(t_w)
        with open('../output/torchID_wiki.json', 'w') as f_json:
            f_json.write(t_w_json)
            print('{}已经生成完毕！'.format(f_json))

    @staticmethod
    def split_data(csv_wiki, image_csv):
        # 导入完整的核心信息表
        """
        核心信息表示例
        ['最小子文件夹名称', '图片张数', '标签数字', '百科条目', '百科ID_来自boss系统', 'PyTorch模型输入输出标签']
        """
        wiki_df = pd.read_csv(csv_wiki)
        image_df = pd.read_csv(image_csv)
        # 按id对应预测结果和真实百科条目
        image_wiki_df = pd.merge(image_df, wiki_df, on='最小子文件夹名称', how='left')
        image_wiki_df = image_wiki_df[['img_path', 'PyTorch模型输入输出标签']]
        # 随机采样验证集的索引
        random.seed(999)
        image_ind = random.sample(range(len(image_wiki_df)), k=len(image_wiki_df))
        train_ind = image_ind[0: int(len(image_ind) * 0.8)]
        val_ind = image_ind[int(len(image_ind) * 0.8): int(len(image_ind) * 0.9)]
        test_ind = image_ind[int(len(image_ind) * 0.9): int(len(image_ind) + 1)]
        l_ind = {'train': train_ind,
                 'val': val_ind,
                 'test': test_ind}
        for k, v in l_ind.items():
            df = image_wiki_df.iloc[v, :]
            out_f = './' + k + '.csv'
            df.to_csv(out_f, index=False, encoding="UTF-8-sig")
            print('包含{}张图片的{}已经生成完毕！'.format(len(df), out_f))
