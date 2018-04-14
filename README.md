# **Behavioral Cloning** 

### The dataset provided by Udacity is used for training the model and simulator provided by udacity to test model.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a single convolution layer with 5x5 filter size and depths 15.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road by adding 0.4 and -0.3 to the steering angle of the that particular image respectively.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive a car through any terrain and any conditions.

My first step was to use a convolution neural network model similar to the NVIDIA I thought this model might be appropriate because it was designed for autonomous vehicles to drive through all kinds of terrains. 

But as compared to realtime scenarios the conditions in the simulation are simpler. So I decided to reduce the model complexity further by reducing convolutional layers and dense layers. This I have done by keeping in mind that the features required by car to drive through road are extracted at initaial levels of convolutional layer and by following this approach I am able to achieve satisfactory results by using 1 convolution layer followed by 1 Dense layer

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I have added dropout layer and maxpooling layer followed by convolution layer and I reduced number of epoches.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I have added left and right side images with offeseted driving angle, I have cropped the image to include only required details so that noise can be reduced, I flipped images so that additional details are available while training the vehicle. 

At the end of the process, the vehicle is able to drive autonomously around both the tracks without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution layer with 5x5 filter having depth 15

Here is a visualization of the architecture 

![alt text](/data/Architecture.png "Architecture")

#### 3. Creation of the Training Set & Training Process

To train my model I used dataset provided by Udacity. Some of the samples from original dataset are

| Camera | Image 1 | Image 2 |
| ---- | ---- | ---- |
| **Center Camera Images** | ![alt text](/data/center1.jpg "Center Images") | ![alt text](/data/center2.jpg "Center Images") |
| **Left Camera Images** | ![alt text](/data/left1.jpg "Left Images") | ![alt text](/data/left2.jpg "Left Images") |
| **Right Camera Images** | ![alt text](/data/right1.jpg "Right Images") | ![alt text](/data/right2.jpg "Right Images") |

First of all I divided data into 80% training data and 20% validation data.

To remove redundant details I cropped image's 80px from top and 20px from bottom for all the images that is center camera images, left camera images and right camera images (right and left camera images are added by adding respective steering angle offset), images after cropping are

| Cropped Center Camera Image | Cropped Left Camera Image | Cropped Right Camera Image |
| ---- | ---- | ----|
| ![alt text](/data/cropped_center.png "Cropped Images") | ![alt text](/data/cropped_left.png "Cropped Images") | ![alt text](/data/cropped_right.png "Cropped Images") |

To augment the data set, I flipped center camera images thinking that this would provide more number of samples to be trained. For example, here is an image that has then been flipped:

| Cropped Center Camera Image | Cropped Flipped Center Camera Image |
| ---- | ---- |
| ![alt text](/data/cropped_center.png "Original Images") | ![alt text](/data/flipped_image.png "Flipped Images") |

After cropping and flipping images I resized all the images to 32 X 32.

| Resized Center Camera Image | Resized Left Camera Image | Resized Right Camera Image |
| ---- | ---- | ----|
| ![alt text](/data/center32.png "Cropped Images") | ![alt text](/data/left32.png "Cropped Images") | ![alt text](/data/right32.png "Cropped Images") |

After cropping image I applied normalization on images and these normalized images are passed to the network as input

| Normalized Center Camera Image | Normalized Left Camera Image | Normalized Right Camera Image |
| ---- | ---- | ----|
| ![alt text](/data/norm_center.png "Cropped Images") | ![alt text](/data/norm_left.png "Cropped Images") | ![alt text](/data/norm_right.png "Cropped Images") |

After the collection process and preprocess, I had 25712 number of training data points. The steering angle distibution for the data points is 

![alt text](/data/processed_images_hist.png "Cropped Images")

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by running the model several times. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Evaluation Video

| Track 1 | Track 2 |
| ---- | ---- |
| [![alt text](/data/track1.jpg "Cropped Images")](https://youtu.be/2HwfRMzT5Mw) | [![alt text](/data/track2.jpg "Cropped Images")](https://youtu.be/TuCbodVgwyc) | 
