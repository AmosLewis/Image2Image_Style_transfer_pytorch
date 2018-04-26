import os
import numpy as np
from PIL import Image


def read_times(stamps_file):
    dic = dict()
    with open(stamps_file) as f:
        for line in f:
            i, d, h, m = line.strip().split(" ")
            dic[i] = int(h)*60 + int(m)
    return dic


def get_pair(time_dic):
    day_start = 11
    day_end = 16
    night_start = 23
    night_end = 4

    day_start = day_start * 60
    day_end = day_end * 60
    night_start = night_start * 60
    night_end = night_end * 60

    day, night = None, None
    for i, t in time_dic.items():
        if t > day_start and t < day_end:
            day = i
        elif t > night_start or t < night_end:
            night = i
        if day is not None and night is not None:
            return day, night
    assert("No suitable image found...")


def reshape_to_array(image, target_size=(32, 32)):
    assert(image.size[0] >= target_size[0] and image.size[1] >= target_size[1])

    # crop
    min_axis = 0 if image.size[0]/target_size[0] < image.size[1]/target_size[1] else 1
    max_axis = 1 - min_axis
    crop_size = [None, None]
    crop_size[min_axis] = image.size[min_axis]
    crop_size[max_axis] = image.size[max_axis]/target_size[max_axis] * target_size[min_axis]
    cropped_image = image.crop((0, 0, crop_size[0], crop_size[1]))

    # resize
    resized_image = cropped_image.resize(target_size)
    resized_image.save("data/temp_path"+str(image.size[0])+"_"+str(image.size[1])+".jpg")
    # to array
    array = np.transpose(np.array(resized_image), (2, 0, 1))

    return array


def extract(img_path, time_path, out_path):
    data = []
    for root, subdirs, files in os.walk(img_path):
        if len(files) > 0 and str.endswith(files[0], ".jpg"):
            stamps = time_path+"/"+root[len(img_path)+1:]+".txt"
            dic = read_times(stamps)
            day, night = get_pair(dic)
            day = Image.open(root+"/"+day)
            night = Image.open(root+"/"+night)
            day = reshape_to_array(day)
            night = reshape_to_array(night)
            print(root, day.shape, night.shape)
            data.append([day, night])
    data = np.asarray(data)
    print("final dataset of shape :", data.shape)
    print("element shapes :", data[0, 0].shape, data[0, 1].shape)
    np.save(out_path, data)


if __name__ == "__main__":

    path_images = "data/dnim/Image"
    path_timestamps = "data/dnim/time_stamp"
    path_output = "data/dnim/dnim.npy"

    extract(path_images, path_timestamps, path_output)
