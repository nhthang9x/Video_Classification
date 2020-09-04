import cv2
import os
import torch
import numpy as np
import csv
import argparse


class Generation():
    def __init__(self, opt):
        super().__init__()

        path = opt.input_path
        img_list = os.listdir(path)
        data = []
        labels = []
        for idx, img_paths in enumerate(img_list):
            content = []
            for (dirpath, dirnames, filenames) in os.walk(os.path.join(path, img_paths)):
                for i, file in enumerate(filenames):
                    image = cv2.imread(filename="{}/{}".format(dirpath, file))
                    content.append(np.array(image / 255))

                if int(img_paths.split(sep="-")[0]) in [3, 9, 12, 27, 29, 30, 32, 39, 40, 43, 44, 45, 55, 65, 68, 73,
                                                        74, 75, 76, 77, 78, 80, 82, 85, 90, 94, 97, 98, 99, 102, 104,
                                                        108, 109, 110, 111, 113, 114, 115]:
                    label = 0
                elif int(img_paths.split(sep="-")[0]) in [4, 6, 7, 10, 15, 16, 17, 20, 33, 34, 36, 49, 50, 63, 64, 66,
                                                          69, 91, 96, 101]:
                    label = 1
                elif int(img_paths.split(sep="-")[0]) in [1, 2, 5, 8, 11, 13, 14, 18, 19, 21, 22, 23, 24, 25, 26, 28,
                                                          31, 35, 37, 38, 41, 42, 46, 47, 48, 51, 52, 53, 54, 56, 57,
                                                          58, 59, 60, 61, 62, 67, 70, 71, 72, 79, 81, 83, 84, 86, 87,
                                                          88, 89, 92, 93, 95, 100, 103, 105, 106, 107, 112]:
                    label = 2
                else:
                    print("Error in Label")

                # data.append(np.array(content))
                labels.append(label)
                content = np.concatenate((content[0], content[1], content[2], content[3], content[4]), axis=2)

            data.append(np.array(content))

        final_data = np.array(data)
        final_label = np.array(labels)

        file = open(opt.data_name, "wb")
        # save array to the file
        np.save(file, final_data)
        # close the file
        file.close

        file = open(opt.label_name, "wb")
        # save array to the file
        np.save(file, final_label)
        # close the file
        file.close

def get_args():
    parser = argparse.ArgumentParser("Pre Processing Video Classification")
    parser.add_argument("-i", "--input_path", type=str, default="output")
    parser.add_argument("-d", "--data_name", type=str, default='train_data')
    parser.add_argument("-l", "--label_name", type=str, default='train_label')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = get_args()
    Generation(opt)