import cv2
import numpy as np
import os
import shutil
import argparse


class Data_Augmentation():
    def __init__(self, opt):
        super().__init__()

        data_path = opt.input_path
        img_path = "{}/".format(opt.output_path)

        if os.path.isdir(opt.output_path):
            shutil.rmtree(opt.output_path)
        os.makedirs(opt.output_path)

        img_paths = os.listdir(data_path)

        for img_file in img_paths:
            element_filename = img_file
            # input(element_filename[:-7])
            base_filename = os.path.join(data_path, element_filename)
            img = cv2.imread(base_filename)
            rows, cols = img.shape[:2]

            img_split = element_filename.strip().split('.jpg')

            # if you specify range(1) total number of augmented data is 6 for one image
            # 6 types of augmentation means pca, horizontal and vertical flip, 3 rotation
            # if you specify range(2) total number of augmented data is 12 for one image
            for color in range(2):
                img_color = self.pca_color_augmentation(img)
                color_name = img_split[0] + '-' + str(color)
                color_jpg = color_name + '.jpg'
                new_path = os.path.join(img_path, color_jpg)
                saved_folder = "{}{}".format(color_jpg[0:-9], color_jpg[-6:-4])
                if not os.path.exists("{}/{}".format(img_path, saved_folder)):
                    os.makedirs("{}/{}".format(img_path, saved_folder))

                new_path = os.path.join(img_path, saved_folder, color_jpg)
                if not os.path.isfile(new_path):
                    cv2.imwrite(new_path, img_color)

                # for horizontal and vertical flip
                f_points = [0, 1]
                f_str = ''
                for f in f_points:
                    f_img = cv2.flip(img_color, f)
                    h, w = img_color.shape[:2]
                    if f == 1:
                        f_str = '2'
                    elif f == 0:
                        f_str = '1'

                    new_name = color_name + '-' + f_str + '.jpg'
                    saved_folder = "{}{}".format(new_name[0:-11], new_name[-8:-4])

                    if not os.path.exists("{}/{}".format(img_path, saved_folder)):
                        os.makedirs("{}/{}".format(img_path, saved_folder))

                    new_path = os.path.join(img_path, saved_folder, new_name)

                    # lines = [new_name, ',', str(f_x1), ',', str(f_y1), ',', str(f_x2), ',', str(f_y2), ',', '\n']
                    # pwd_lines.append(lines)
                    if not os.path.isfile(new_path):
                        cv2.imwrite(new_path, f_img)

                # for angle 90
                img_transpose = np.transpose(img_color, (1, 0, 2))
                img_90 = cv2.flip(img_transpose, 1)
                h, w = img_color.shape[:2]
                new_name = color_name + '-' + '3' + '.jpg'
                saved_folder = "{}{}".format(new_name[0:-11], new_name[-8:-4])

                if not os.path.exists("{}/{}".format(img_path, saved_folder)):
                    os.makedirs("{}/{}".format(img_path, saved_folder))

                new_path = os.path.join(img_path, saved_folder, new_name)
                if not os.path.isfile(new_path):
                    cv2.imwrite(new_path, img_90)

                # for angle 180
                img_180 = cv2.flip(img_color, -1)
                new_name = color_name + '-' + '4' + '.jpg'
                # print(new_name)
                saved_folder = "{}{}".format(new_name[0:-11], new_name[-8:-4])

                if not os.path.exists("{}/{}".format(img_path, saved_folder)):
                    os.makedirs("{}/{}".format(img_path, saved_folder))

                new_path = os.path.join(img_path, saved_folder, new_name)
                if not os.path.isfile(new_path):
                    cv2.imwrite(new_path, img_180)

                # for angle 270
                img_transpose_270 = np.transpose(img_color, (1, 0, 2))
                img_270 = cv2.flip(img_transpose_270, 0)
                new_name = color_name + '-' + '5' + '.jpg'
                saved_folder = "{}{}".format(new_name[0:-11], new_name[-8:-4])

                if not os.path.exists("{}/{}".format(img_path, saved_folder)):
                    os.makedirs("{}/{}".format(img_path, saved_folder))

                new_path = os.path.join(img_path, saved_folder, new_name)
                if not os.path.isfile(new_path):
                    cv2.imwrite(new_path, img_270)


    def pca_color_augmentation(self, image):
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.dtype == np.uint8
        img = image.reshape(-1, 3).astype(np.float32)
        sf = np.sqrt(3.0 / np.sum(np.var(img, axis=0)))
        img = (img - np.mean(img, axis=0)) * sf
        cov = np.cov(img, rowvar=False)  # calculate the covariance matrix
        # calculation of eigen vector and eigen value
        value, p = np.linalg.eig(cov)
        rand = np.random.randn(3) * 0.08
        delta = np.dot(p, rand * value)
        delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]
        img_out = np.clip(image + delta, 0, 255).astype(np.uint8)
        return img_out


def get_args():
    parser = argparse.ArgumentParser("Pre Processing Video Classification")
    parser.add_argument("-i", "--input_path", type=str, default="cropped_data")
    parser.add_argument("-o", "--output_path", type=str, default='output')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    Data_Augmentation(opt)
