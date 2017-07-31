# Writeup

The Python code can be found in [IPython notebook](./vehicle_detector.ipynb).

---

**Vehicle Detection Project**

The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[test_image]: ./test_images/test1.jpg

[boxes1]: ./output_images/boxes_1.jpg
[boxes2]: ./output_images/boxes_2.jpg
[boxes3]: ./output_images/boxes_3.jpg

[cars_found1]: ./output_images/cars_found1.jpg
[cars_found2]: ./output_images/cars_found2.jpg
[cars_found3]: ./output_images/cars_found3.jpg
[cars_found4]: ./output_images/cars_found4.jpg
[cars_found5]: ./output_images/cars_found5.jpg
[cars_found6]: ./output_images/cars_found6.jpg

[heat1]: ./output_images/heat1.jpg
[heat2]: ./output_images/heat2.jpg
[heat3]: ./output_images/heat3.jpg
[heat4]: ./output_images/heat4.jpg
[heat5]: ./output_images/heat5.jpg
[heat6]: ./output_images/heat6.jpg

[hog_car]: ./output_images/hog_car.jpg
[hog_notcar]: ./output_images/hog_notcar.jpg

[car]: ./output_images/car.png
[notcar]: ./output_images/notcar.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---

### Features extraction (HOG and colors)


The code for this step is contained in the `Histograms of oriented gradients` code cell of the [IPython notebook](./vehicle_detector.ipynb).  

The datasets of GTI and KITTI `vehicles` and `non-vehicles` images were used (8792 images of vehicles and 8968 images of non-vehicles). Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![][car]
![][notcar]

Then different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Random images were grabbed from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG
`YCrCb` color space (all channels were considered), parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for both car and non-car images:

![][hog_car]
![][hog_notcar]

Various combinations of parameters were tried and it appears that best fit for the current videos pipelines are
`YCrCb` color space, `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. For example, when setting `orientations=12`, classifier test accuracy stays almost the same, but model becomes overfitted and runs poor on real images.
Also, histograms of the `YCrCb` color layers (histogram bins = `16`) were computed and spatially binned color features (spatial size of `16x16`) were appended as well.
All features extraction used in the classifier is described in `Features extraction` code cell (`extract_features()` function) in the [IPython notebook](./vehicle_detector.ipynb).

The Linear Support Vector Machine classifier was trained on the given data (`Support Vector Machine classifier` code cell). The features data were scaled to the zero mean and unit variance first before feeding into classifier. To perform this, the `StandardScaler` of `sklearn` framework was used (`Scaling the data` code cell). The data was then split into train and test sets (test set is 20% of overall count). The Linear SVM was trained on the train set. It gives about 0.9907 accuracy on the test one.

### Sliding Window Search

I decided to use three different scales (`1.0`, `1.2`, `1.5`) at Y start-stop positions of `[400, 480]`, `[400, 580]`, `[400, 625]` accordingly. HOG features were calculated only once for every scale. The windows were used of size `64x64` with `cells_per_step=2` (which results in overlapping of `75%`). The search is implemented in `find_cars_boxes()` function in the `Sliding window search` code cell.

The results of applying window search with different scales are the following:
![][test_image]
![][boxes1]
![][boxes2]
![][boxes3]

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.

Based on the found windows the heatmap was created and the map was then thresholded to get rid of possible false positives (`find_cars()` function in the [notebook](./vehicle_detector.ipynb)).

The examples of heatmaps are the following:
![][heat1]
![][heat2]
![][heat3]
![][heat4]
![][heat5]
![][heat6]

`scipy.ndimage.measurements.label()` was used to identify individual blobs in the heatmap. So the final cars boxes were found:

![][cars_found1]
![][cars_found2]
![][cars_found3]
![][cars_found4]
![][cars_found5]
![][cars_found6]

---

### Video Implementation

Here is [the project video](./output_videos/output_project_video.mp4) processed by the pipeline.

The heatmaps of each frame of the video were recorded. Then they were combined for last 5 video frames and then the resulted sum was thresholded to get rid of possible false positives over the considered frames. The pipeline is coded in `find_cars_video()` function in the [notebook](./vehicle_detector.ipynb).

---

### Discussion

The resulted pipeline doesn't recognize vehicles shot from the front because of the lack of the training data. Also false positives still appear (though they are managed to be thresholded out via heatmaps). So first of all further improvements should include more data for classifier to be trained on. It will also allow to add more features from the extraction (more HOG orientations, considering of other color spaces as well etc.) without model overfitting so the detection becomes more accurate.
