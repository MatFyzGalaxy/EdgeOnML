import cv2
import os
import numpy as np

whole_galaxies_path = 'data\images'
labels_path = 'data\labels'
masks_path = 'data\galaxies_masks'
dest_path = 'data\image_masks'


def paste_image(coords: str, img: np.ndarray, image_name: str, mask: np.ndarray):
    dh, dw = img.shape

    box = coords
    class_id, x_center, y_center, w, h, _ = box.strip().split()
    x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
    x_center = round(x_center * dw)
    y_center = round(y_center * dh)
    w = round(w * dw)
    h = round(h * dh)

    x = round(x_center - w / 2)
    y = round(y_center - h / 2)

    img[y:y + mask.shape[0], x:x + mask.shape[1]] = mask

    # Saving the image
    cv2.imwrite(os.path.join(dest_path, image_name), img)


def paste_image_and_save():
    masks = os.listdir(masks_path)
    print("Processing images...")
    missing_images = []
    missing_coords = []

    for mask in masks:
        print(mask)
        mask_image = cv2.imread(fr"{masks_path}\{mask}")
        # deleting 3rd dimension of mask
        mask_image = mask_image[:, :, 0]

        mask_name, _, rank = mask.split(".")[0].split(' ')
        rank = int(rank)
        print(f"{mask}: {mask_name}, {rank}")
        img = cv2.imread(fr"{whole_galaxies_path}\{mask_name}.jpg", 0)

        # painting image to black
        try:
            img *= 0
        except TypeError:
            missing_images.append(mask_name)

        try:
            file = open(fr"{labels_path}\{mask_name}.txt", 'r')
            coords = file.readlines()[rank]

            paste_image(coords, img, mask, mask_image)
            file.close()
        except FileNotFoundError:
            missing_coords.append(mask_name)

    print("Printing missing images:")
    for image in missing_images:
        print(image)

    print("Printing missing coordinates:")
    for coords in missing_coords:
        print(coords)


paste_image_and_save()
