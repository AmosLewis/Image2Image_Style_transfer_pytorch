import os
import numpy as np
import pandas as pd
from PIL import Image


def extract_image(img_path):
    img = Image.open(img_path)
    img = img.crop((0, 0, 512, 256))
    array = np.transpose(np.array(img), (2, 0, 1))
    print(img_path, array.shape)
    return array


def extract(path_images_day, path_images_night, path_matches, path_output):

    matches = pd.read_csv(path_matches).as_matrix()
    print(matches.shape)
    matches = matches[:len(matches) // 2]
    print(matches)
    max_img = matches.max()

    day = []
    day_imgs = sorted(os.listdir(path_images_day))
    for img in day_imgs[:max_img+1]:
        day.append(extract_image(path_images_day+"/"+img))
    day = np.array(day)

    night = []
    night_imgs = sorted(os.listdir(path_images_night))
    for img in night_imgs[:max_img + 1]:
        night.append(extract_image(path_images_night+"/"+img))
    night = np.array(night)

    matches = matches.T

    day = np.expand_dims(day[matches[0]], axis=1)
    night = np.expand_dims(night[matches[1]], axis=1)
    data = np.concatenate([day, night], axis=1)
    print(data.shape)

    np.save(path_output, data)


if __name__ == "__main__":
    prefix = "data/alderley/"
    path_images_day = prefix+"FRAMESA"
    path_images_night = prefix+"FRAMESB"
    path_matches = prefix+"framematches.csv"
    path_output = prefix+"alderley.npy"

    extract(path_images_day, path_images_night, path_matches, path_output)
