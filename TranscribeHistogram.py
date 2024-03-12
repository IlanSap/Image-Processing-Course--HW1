import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore")


# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
    img_size = imgs_arr[0].shape
    res = []

    for img in imgs_arr:
        X = img.reshape(img_size[0] * img_size[1], 1)
        km = KMeans(n_clusters=n_colors)
        km.fit(X)

        img_compressed = km.cluster_centers_[km.labels_]
        img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

        res.append(img_compressed.reshape(img_size[0], img_size[1]))

    return np.array(res)


# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
    image_arrays = []
    lst = [file for file in os.listdir(folder) if file.endswith(formats)]
    for filename in lst:
        file_path = os.path.join(folder, filename) # create the full path to an image
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_arrays.append(gray_image)
    return np.array(image_arrays), lst


# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
    # Assuming the image is of the same pixel proportions as images supplied in this exercise,
    # the following values will work
    x_pos = 70 + 40 * idx
    y_pos = 274
    while image[y_pos, x_pos] == 0:
        y_pos -= 1
    return 274 - y_pos

# input: image, numpy array
# output: the function creates the pixel's histogram of the image,
# and then calculates the accumulated histogram and returns it.
def calcAccumulatedHist(image):

    window_hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    window_accumulated_histogram = np.zeros(256)
    for i in range(256):
        if (i == 0):
            window_accumulated_histogram[i] = window_hist[i]
        else:
            window_accumulated_histogram[i] = window_accumulated_histogram[i - 1] + window_hist[i]
    return window_accumulated_histogram

# input: src_image- numpy array of grade histogram picture and target- numpy array of number image
# output: the function divides src_image to slide windows size height X width and search in specific area in
# src_image the window_slide with EMD<260.
# The function returns True if a sliding window with EMD<260 was found, False otherwise.
def compare_hist(src_image, target):

    height = 15
    width = 10

    target_cumulative_histogram = calcAccumulatedHist(target)
    windows = np.lib.stride_tricks.sliding_window_view(src_image, (height, width))
    for x in range(25, 50):
        for y in range(110, 130):

            current_window = windows[y, x]
            window_cumulative_histogram = calcAccumulatedHist(current_window)
            result_array = np.abs(window_cumulative_histogram - target_cumulative_histogram)
            emd = np.sum(result_array)

            if emd < 260: return True

    return False
# input: an image of grade histogram and max number in the vertical axis
# Calculate the number of students per bin in a histogram image.
# This function estimates the number of students in each bin of the histogram
# based on the height of the bars in the image
# output: number of students per bin in the image
def calsBinsHeight(image, max_number):
    bins_height_in_pixels = []
    students_per_bin = []
    for binId in range(10):
        bins_height_in_pixels.append(get_bar_height(image, binId))
    for binId in range(10):
        students_per_bin.append(np.round(max_number * bins_height_in_pixels[binId] / max(bins_height_in_pixels)))
    return students_per_bin

# input: images (numpy array): Array of images
#  This function converts each pixel in the images to either black (0) or
#     white (1) based on a threshold value.
# output:  numpy array: Array of black and white images. (only 0,1 value)
def calcThresholdImages(images):
    thresholdImages = []
    for image in images:
        # we searched a coordinate of points in a bright gray bin in every image and make the max value + margin of safety (+5) as the threshold
        thresholdValue = 220
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                if image[y][x] <= thresholdValue:
                    image[y][x] = 0
                else:
                    image[y][x] = 1
        thresholdImages.append(image)
    return np.array(thresholdImages)

# main
images, names = read_dir('data')
numbers, numberNames = read_dir('numbers')
quantizedImages = quantization(images,3)

thresholdImages = calcThresholdImages(quantizedImages)

indexOfImages=0
for image in images:
    # The loop runs on the numbers images from the last image to the first.
    for i in range(len(numbers)-1, -1, -1):
        if compare_hist(image, numbers[i]):
            break

    number_file_name = numberNames[i]

    students_per_bin = calsBinsHeight(thresholdImages[indexOfImages], int(number_file_name[0]))
    print(f'Histogram {names[indexOfImages]} gave {",".join(map(str, map(int, students_per_bin)))}')

    # # Plotting the students_per_bin
    # plt.figure(figsize=(8, 5))
    # plt.bar(range(10), students_per_bin, color='skyblue')
    # plt.xlabel('Bin Index')
    # plt.ylabel('Number of Students')
    # plt.title('Number of Students per Bin')
    # plt.xticks(range(10))
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    #
    # # Plotting the window_accumulated_histogram
    # window_accumulated_histogram = calcAccumulatedHist(image)
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(256), window_accumulated_histogram, color='orange')
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Cumulative Frequency')
    # plt.title('Window Accumulated Histogram')
    # plt.grid(linestyle='--', alpha=0.7)
    # plt.show()

    indexOfImages += 1

exit()