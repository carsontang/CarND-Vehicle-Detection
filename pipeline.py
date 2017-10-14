import glob
import pickle
import time
from collections import deque

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

import global_settings


class RingBuffer(object):
    """
    Buffer N elements, each a list
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque()

    def add(self, a_list):
        self.buffer.append(a_list)

        if len(self.buffer) >= self.capacity:
            self.buffer.popleft()

    def get_all_lists(self):
        results = []
        for element in self.buffer:
            results = results + element
        return results

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
            visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
            visualise=False, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel()

def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def extract_features(images, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     use_spatial_feat=True, use_hist_feat=True, use_hog_feat=True):
    """
    Can extract up to 3 types of features:
    1) pixel intensity values from scaled down images
    2) histogram of pixel intensities
    3) histogram of gradients
    """

    features = []
    for img in tqdm(images):
        img_features = single_img_features(img, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                           orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                           hog_channel=hog_channel, use_spatial_feat=use_spatial_feat,
                                           use_hist_feat=use_hist_feat, use_hog_feat=use_hog_feat)
        features.append(img_features)
    return features

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        use_spatial_feat=True, use_hist_feat=True, use_hog_feat=True):

    features = []

    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif color_space == 'GRAY':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                feature_image = np.stack((feature_image, feature_image, feature_image), axis=2)  # keep shape
    else:
        feature_image = np.copy(img)

    if use_spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        features.append(spatial_features)

    if use_hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        features.append(hist_features)

    if use_hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        features.append(hog_features)


    return np.concatenate(features)

def train(cars, notcars, svc, X_scaler):
    car_features = extract_features(cars, color_space=global_settings.color_space,
                                    spatial_size=global_settings.spatial_size, hist_bins=global_settings.hist_bins,
                                    orient=global_settings.orient, pix_per_cell=global_settings.pix_per_cell,
                                    cell_per_block=global_settings.cell_per_block,
                                    hog_channel=global_settings.hog_channel, use_spatial_feat=global_settings.use_spatial_feat,
                                    use_hist_feat=global_settings.use_hist_feat, use_hog_feat=global_settings.use_hog_feat)
    notcar_features = extract_features(notcars, color_space=global_settings.color_space,
                                       spatial_size=global_settings.spatial_size, hist_bins=global_settings.hist_bins,
                                       orient=global_settings.orient, pix_per_cell=global_settings.pix_per_cell,
                                       cell_per_block=global_settings.cell_per_block,
                                       hog_channel=global_settings.hog_channel, use_spatial_feat=global_settings.use_spatial_feat,
                                       use_hist_feat=global_settings.use_hist_feat, use_hog_feat=global_settings.use_hog_feat)

    X = np.vstack((car_features, notcar_features, notcar_features, notcar_features)).astype(np.float64)
    X_scaler.fit(X)
    scaled_X = X_scaler.transform(X)

    # '0' is for non-cars
    # '1' is for cars
    y = np.hstack((np.ones(len(car_features)), np.zeros(3*len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.05, random_state=rand_state)

    print('Settings: orient=%(orient)d, pix_per_cell=%(pix_per_cell)d, cell_per_block=%(cell_per_block)d'
          % { 'orient': global_settings.orient,
              'pix_per_cell': global_settings.pix_per_cell,
              'cell_per_block': global_settings.cell_per_block})
    print('Feature vector length:', len(X_train[0]))

    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, use_spatial_feat=spatial_feat,
                                       use_hist_feat=hist_feat, use_hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Add heat to heatmap
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


# Apply threshold to heat map
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def numpyify(pattern):
    arrs = []
    for imgfile in glob.glob(pattern):
        img = mpimg.imread(imgfile)
        arrs.append(img)
    return arrs

def demo_windows():
    svc = LinearSVC()
    X_scaler = StandardScaler()

    if global_settings.use_pretrained_model:
        print("Using pretrained model...")
        with open('model.p', 'rb') as f:
            d = pickle.load(f)
        svc = d['svc']
        X_scaler = d['X_scaler']
        print('Loaded pre-trained model from model.p')
    else:
        print('Loading dataset of cars and non-cars')
        cars = numpyify('dataset/vehicles/*/*')
        notcars = numpyify('dataset/non-vehicles/*/*')

        print('Training classifier')
        train(cars, notcars, svc, X_scaler)

        print('Training completed, and serializing trained model')
        with open('model.p', 'wb') as f:
            pickle.dump({'svc': svc, 'X_scaler': X_scaler}, f)

    image_file = glob.glob('test_images/*.jpg')[0]
    image = mpimg.imread(image_file)
    draw_image = np.copy(image)

    windows = slide_window(image, x_start_stop=(100, 1200), y_start_stop=(400, 600),
                           xy_window=(96, 96), xy_overlap=(global_settings.pct_overlap, global_settings.pct_overlap))

    bboxes = search_windows(image, windows, svc, X_scaler, color_space=global_settings.color_space,
                                 spatial_size=global_settings.spatial_size, hist_bins=global_settings.hist_bins,
                                 orient=global_settings.orient, pix_per_cell=global_settings.pix_per_cell,
                                 cell_per_block=global_settings.cell_per_block,
                                 hog_channel=global_settings.hog_channel, spatial_feat=global_settings.use_spatial_feat,
                                 hist_feat=global_settings.use_hist_feat, hog_feat=global_settings.use_hog_feat)

    multiwindowed_img = draw_boxes(draw_image, bboxes, color=(0, 0, 255), thick=6)

    plt.imshow(multiwindowed_img)
    plt.show()

    heatmap = np.zeros(image.shape[:2])
    heatmap = add_heat(heatmap, bboxes)
    heatmap = apply_threshold(heatmap, 1)
    labels = label(heatmap)
    plt.imshow(labels[0], cmap='gray')
    plt.show()

    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    plt.imshow(draw_img)
    plt.show()

window_buffer = RingBuffer(global_settings.ring_buffer_cap)
svc = LinearSVC()
X_scaler = StandardScaler()

def annotate_image(image):
    global window_buffer, svc, X_scaler

    windows = slide_window(image, x_start_stop=(100, 1200), y_start_stop=(400, 600),
                        xy_window=(96, 96), xy_overlap=(global_settings.pct_overlap, global_settings.pct_overlap))

    new_windows = search_windows(image, windows, svc, X_scaler, color_space=global_settings.color_space,
                                     spatial_size=global_settings.spatial_size, hist_bins=global_settings.hist_bins,
                                     orient=global_settings.orient, pix_per_cell=global_settings.pix_per_cell,
                                     cell_per_block=global_settings.cell_per_block,
                                     hog_channel=global_settings.hog_channel, spatial_feat=global_settings.use_spatial_feat,
                                     hist_feat=global_settings.use_hist_feat, hog_feat=global_settings.use_hog_feat)

    # Add new windows to ring buffer
    window_buffer.add(new_windows)
    buffer_windows = window_buffer.get_all_lists()

    # Calculate and draw heat map
    heatmap = np.zeros(image.shape[:2])
    heatmap = add_heat(heatmap, buffer_windows)
    heatmap = apply_threshold(heatmap, global_settings.heatmap_threshold)
    labels = label(heatmap)

    # Draw final bounding boxes
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img

def annotate_video(input_file, output_file):
    global window_buffer, svc, X_scaler

    with open('model.p', 'rb') as f:
        save_dict = pickle.load(f)
    svc = save_dict['svc']
    X_scaler = save_dict['X_scaler']
    print('Loaded pre-trained model from model.p')

    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(annotate_image)
    annotated_video.write_videofile(output_file, audio=False)

if __name__ == '__main__':
    annotate_video('project_video.mp4', 'output.mp4')