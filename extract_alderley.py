import os
import numpy as np
import pandas as pd
from PIL import Image


def extract_image(img_path):
    img = Image.open(img_path)
    img = img.crop((0, 0, 512, 256))
    array = np.transpose(np.array(img), (2, 0, 1))/255
    print(img_path, array.shape)
    return array.astype(np.float16)


def extract(path_images_day, path_images_night, path_matches, path_output):

    matches = pd.read_csv(path_matches).as_matrix()
    print(matches.shape)

    a = np.arange(0, len(matches), 50)
    matches = matches[a].T
    print(matches)

    day = []
    day_imgs = np.array(sorted(os.listdir(path_images_day)))

    for img in day_imgs[matches[1]-1]:
        day.append(extract_image(path_images_day+"/"+img))
    day = np.array(day)

    night = []
    night_imgs = np.array(sorted(os.listdir(path_images_night)))
    for img in night_imgs[matches[0]-1]:
        night.append(extract_image(path_images_night+"/"+img))
    night = np.array(night)

    day = np.expand_dims(day, axis=1)
    night = np.expand_dims(night, axis=1)
    data = np.concatenate([day, night], axis=1)
    print(data.shape)

    np.save(path_output, data)


if __name__ == "__main__":
    prefix = "data/alderley/"
    path_images_day = prefix+"FRAMESB"
    path_images_night = prefix+"FRAMESA"
    path_matches = prefix+"framematches.csv"
    path_output = prefix+"alderley.npy"

    extract(path_images_day, path_images_night, path_matches, path_output)
