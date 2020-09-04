import cv2
import os
import shutil
import argparse


class Preprocess():
    def __init__(self, opt):
        super().__init__()

        if os.path.isdir('cropped_data'):
            shutil.rmtree('cropped_data')
        os.makedirs('cropped_data')

        for i in range(opt.first_value, opt.second_value):
            for j in range(51, 61):
                if j in [52, 54, 56, 58, 60]:
                    image = cv2.imread(filename="all/norm_film{}_frame{}.jpg".format(i, j))
                    image = image[50:570, 170:690]
                    cv2.imwrite(filename="cropped_data/{}-{}.jpg".format(i, j), img=image)

def get_args():
    parser = argparse.ArgumentParser("Pre Processing Video Classification")
    parser.add_argument("-f", "--first_value", type=int, default=1)
    parser.add_argument("-s", "--second_value", type=int, default=101)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    Preprocess(opt)
