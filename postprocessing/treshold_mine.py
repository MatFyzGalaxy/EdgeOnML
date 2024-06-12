import cv2
import os
import numpy as np

# path to input image is specified
image_path = 'data\galaxies'
path = 'data\galaxies_masks'

images = os.listdir(image_path)

# image is loaded with imread command
for image_file in images:
    image_name = image_file.split(".")[0]
    image = cv2.imread(fr"{image_path}\{image_name}.jpg")

    # cv2.cvtColor is applied over the
    # image input with applied parameters
    # to convert the image in grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh, im_bw = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Thresh: ", thresh)
    # applying different thresholding
    # techniques on the input image
    # all pixels value above 120 will
    # be set to 255
    #ret, image_mask = cv2.threshold(image_gray, thresh, 255, cv2.THRESH_BINARY)
    #print("Thresh: ", ret)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(im_bw, kernel, iterations=2)
    mask = cv2.erode(im_bw, kernel, iterations=2)
    #mask = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)

    # the window showing output images
    # with the corresponding thresholding
    # techniques applied to the input images
    # cv2.imwrite("thresh_mine.jpg",thresh1)
    cv2.imwrite(os.path.join(path, f"{image_name}.jpg"), mask)

    # cv2.imshow('Binary Threshold', thresh1)
    # cv2.imshow('Binary Threshold', im_bw)

    # De-allocate any associated memory usage
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
