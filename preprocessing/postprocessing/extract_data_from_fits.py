import os

import cv2
import numpy as np
from PIL import Image
from astropy.io import fits

# Define the paths to the FITS file and the mask in JPG format
FITS_DIR = 'data\fits'
MASKS_DIR = 'data\image_masks'
OUTPUT_DIR = 'data\galaxies_fits'
FITS_FILENAMES = os.listdir(FITS_DIR)
MASK_FILENAMES = os.listdir(MASKS_DIR)

for mask_filename in MASK_FILENAMES:
    # Open the FITS file
    with fits.open(fr"{FITS_DIR}\{mask_filename.split(' ')[0]}.fit.gz") as hdul:
        # Access the data array of the primary HDU (assuming it's the first extension)
        data = hdul[0].data

        # Open the mask image
        mask_image = Image.open(fr"{MASKS_DIR}\{mask_filename}")
        # mask_image = cv2.imread(mask_file_path)
        desired_width = data.shape[1]
        desired_height = data.shape[0]

        # Resize the mask image while maintaining the original shape
        resized_mask = mask_image.resize((desired_width, desired_height), resample=Image.NEAREST)
        resized_mask.save("tmp.jpg")

        resized_mask = cv2.imread("tmp.jpg")
        resized_mask = resized_mask[:, :, 0]

        resized_mask = resized_mask[::-1]
        bin_mask = (resized_mask < 50)
        new_image = np.copy(data)
        new_image[bin_mask] = resized_mask[bin_mask]

        # Create a new FITS file to save the extracted data
        # hdu = fits.PrimaryHDU(new_image)
        # hdul_new = fits.HDUList([hdu])
        hdul_new = hdul
        hdul_new[0].data = new_image

        # Save the extracted data to a new FITS file
        # output_file_path = 'fpC-005194-g5-0367_extracted.fit'
        hdul_new.writeto(fr"{OUTPUT_DIR}\{mask_filename.split('.')[0]}.fit.gz", overwrite=True, output_verify='ignore')

        # ### show before with aplpy
        # gc = aplpy.FITSFigure(fr"{FITS_DIR}\{mask_filename.split(' ')[0]}.fit.gz")
        # gc.show_grayscale(invert=False, stretch='power', exponent=0.5)
        # plt.show()
        #
        # ### show result with aplpy
        # gc = aplpy.FITSFigure(fr"{OUTPUT_DIR}\{mask_filename.split('.')[0]}.fit.gz")
        # gc.show_grayscale(invert=False, stretch='power', exponent=0.5)
        # plt.show()
