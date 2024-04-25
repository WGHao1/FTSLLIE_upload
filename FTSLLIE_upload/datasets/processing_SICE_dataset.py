import os
import random
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
from natsort import os_sorted
# -*- coding: UTF-8 -*-
import os
import imageio
import shutil

#确定文件夹中文件数量
def file_count(file_path):
    count = 0
    for file in os.listdir(file_path):  # file 表示的是文件名
        count = count + 1
    print(file_path, "中有多少文件：", count)
    return count

#按文件顺序读取
def file_order(file_path):
    fileNames = os.listdir(file_path)#制作成列表
    file_list = os_sorted(fileNames)#对列表进行排序
    image_count1 = file_count(file_path)#统计数量
    # num = 0
    # for file in file_list:
    #     file_name = file_path + "/" + file
        # if (num == 0 or num == 1 or num == image_count1 or num == image_count1 / 2 or num == (image_count1 / 2) + 1):
        #     print('#仅作部分展示，判断程序是否正常:')
        #     print(file_name)
        #     print('#仅作部分展示，判断程序是否正常:')
        #     #仅作部分展示，判断程序是否正常
        # num += 1
    return file_list

#按文件顺序读取SICE中的所有图片
def image_order(file_path):
    file_list = file_order(file_path)
    image_list = []
    for file in file_list:
        file_name = file_path + "/" + file
        image_list = image_list + file_order(file_name)

    # print('image_list:',image_list)
    # #仅作部分展示，判断程序是否正常
    return image_list


# 把SICE数据集 各个子文件夹中的图片重命名之后全部复制到train或者eval文件里面
def train_root(file_path):
    # 打开文件

    path = file_path
    image_order(path)
    if path == '../data/SICE/Dataset_Part1':
        newFilePath = "../data/SICE/train"
    elif path == '../data/SICE/Dataset_Part2(1)':
        newFilePath = "../data/SICE/eval"
    else:
        print('未检测到SICE数据集：../data/SICE/Dataset_Part1 和 ../data/SICE/Dataset_Part2(1)')
    dirs = os.listdir(path)
    print(dirs)  # 输出所有子文件和文件夹

    for file in dirs:
        pic_dir = os.path.join(path, file)  # images中子文件夹的路径
        num = 1
        for i in os.listdir(pic_dir):

            src = os.path.join(os.path.abspath(pic_dir), i)# 旧的路径
            dst = os.path.join(os.path.abspath(newFilePath), i)# 新的路径

            new_name = os.path.join(os.path.abspath(newFilePath), os.path.basename(pic_dir) + '_' + str(num) + '.JPG')

            num = num + 1

            # 复制图像
            shutil.copy(src, dst)

            # 重命名
            os.rename(dst, new_name)

            print(src)
            print(new_name)

    #确定有多少张图片
    path1 = newFilePath
    train_Bacterialblight = 0
    for Bacterialblight in os.listdir(path1):
        train_Bacterialblight = train_Bacterialblight + 1
    print("train_Bacterialblight:" + str(train_Bacterialblight) + "张")
    # 确定有多少张图片

    return newFilePath


if __name__ == '__main__':
    train_data = '../data/SICE/Dataset_Part1'
    train_root(train_data)
    val_data = '../data/SICE/Dataset_Part2(1)'
    train_root(val_data)


