import cv2
import os
import numpy as np

image_path = 'data\images'
labels_path = 'data\labels'
path = 'data\galaxies'


def crop_image(coords: str, img: np.ndarray, image_name: str):
    dh, dw, _ = img.shape

    box = coords
    _, x_center, y_center, w, h, _ = box.strip().split()
    x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
    x_center = round(x_center * dw)
    y_center = round(y_center * dh)
    w = round(w * dw)
    h = round(h * dh)
    x = round(x_center - w / 2)
    y = round(y_center - h / 2)

    cropped_image = img[y:y + h, x:x + w]

    # Saving the image
    cv2.imwrite(os.path.join(path, image_name), cropped_image)


def crop_and_save():
    labels = os.listdir(labels_path)
    print("Processing images...")

    for label in labels:
        label_name = label.split('.')[0]

        img = cv2.imread(fr"{image_path}\{label_name}.jpg")

        file = open(fr"{labels_path}\{label}", 'r')
        lines = file.readlines()
        for i, line in enumerate(lines):
            crop_image(line, img, f"{label_name} - {i}.jpg")


crop_and_save()
