# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results
* run_max folder contains the Autonomous driving images of run on Simulator with my model and run.mp4 is the video generated from this run images

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

I have used the dataset provided by Udacity in the "./data/IMG" folder, as it contains multiple laps of driving and also recovery dataset from side of road. It contains images from 3 cameras: right, left and centre for same frame. It worked well for my model training and so I decided to continue with it. "./data/driving_log.csv" contains steering angles for these images in data set. The loading of dataset images can be found in lines 15-21 of model.py file. I have read the images from Dataset in RGB format using "ndimage.imread(current_path)", using ndimage from Scipy. This can be found at line 44 of model.py file.

The dataset contains images from Centre, Left and Right Camera. The Images from Left and Right Cameras are biased towards side as compared to centre image and can be seen below:

[left]:./Images/left.PNG "Left Camera Image"

![Left][left]

[centre]:./Images/centre.PNG "Centre Camera Image"

![Centre][centre]

[right]:./Images/right.PNG "Right Camera Image"

![Right][right]

I have introduced a correction factor of 0.2 to the steering of left and right camera images to correct their Steering angles. This can be found at lines 47-54 of model.py file.

Data Augmentation: I have Flipped the images in the dataset around vertical axis. This gives me dataset of double size than the existing one. Also, the steering angle is reversed by multiplying with (-1.0) to get correct steering value for these flipped images. This can be found at lines 58-65 in model.py file.

I have divided the datset to take 20% of it as Validation data and rest 80% as Training data. This can be found at line 23-24 of model.py. Also I have shuffled the images in dataset so that the images gets evenly distributed between Validation and Training sets. This can be found at line 31 of model.py.

Model Architecture can be found at lines 107-165 of model.py.
I started by using Nvidia autonomous model. Its architecture has been explained [here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The model contains 3 Convolution layers of kernel_size=5,5 and strides=2,; 2 Convolution layers of kernel size=3,3 ;and 3 Dense layers.

Firstly, its necessary to normalize the input images when making prediction with drive.py. This has been taken care by beginning the model architecture with Lambda layer at line 109 of model.py.

Now, as can be seen in all Camera images used for training, the top ~70 pixels contain Trees and Sky and bottom ~25 pixels contains Car's bonnet and are not much useful for the Car to drive. So its better to remove the unwanted part of images, to reduced the space and increase efficiency of model. For this I have used a Cropping2D layer. It can be seen at line 111 of model.py. I have removed top 70 and bottom 25 pixels with this Croppoing layer. The cropped image should look something like this:

[cropped]:./Images/cropped.png "Cropped Image"

![Cropped][cropped]

In order to reduce Model's overfitting, I introduced a Dropout Layer in Nvidia's Autonomous model with 0.25 rate.

I was facing error "ValueError: Error when checking target: expected dense_4 to have 4 dimensions, but got array with shape (192, 1)". After googling the reason for this error, I got to know that the output of Convolution layers is of 4 dimensions but the further Dense layers need 2 dimension, so I need to use a Flatten layer to convert from 4 dimension to 2 dimension. I added a Flatten layer before the Dense layer in my model and can be seen at line 160 of model.py.

This model can be found as "model_nv.h5" in the same folder and corresponding run on Simulator Track1 can be found as "run_nv.mp4" in same folder. It worked well for Track1 in simulator but couldn't travel well for Track2 in simulator. So I started exploring Convolution layers with different Activation functions in the same model architecture. Some of the models created by me are as shown below:

1) I used RELU activation with Convolutional2D layer. Within 5 Epoch of model training it could achieve Training loss: 0.0146 and validation loss: 0.0152. This model can be found as model_relu.h5 in the same folder. Its summary is as below:

Layer (type)                 Output Shape              Param #   

=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11       

=================================================================

Total params: 348,219

Trainable params: 348,219

Non-trainable params: 0
_________________________________________________________________

2) Next I used ELU activation function instead of RELU activation function. With this architecture I was able to achieve training loss: 0.0195 - val_loss: 0.0213. Its model can be found as "model_elu.h5" in the same folder. Its summary is as below:

Layer (type)                 Output Shape              Param #   

=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        

=================================================================

Total params: 348,219

Trainable params: 348,219

Non-trainable params: 0
_________________________________________________________________
This wasn't much better than previous architecture explained in #1, so I moved on to explore model training architecture further.

3) Next I came across Swish activation funcion from (this site)[https://www.bignerdranch.com/blog/implementing-swish-activation-function-in-keras/]. I heard that is performs better than RELU activation function and so tried training my model using it in architecture. It can be found as "model_swish.h5" in same folder. Its summary is as below:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   

=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        

=================================================================

Total params: 348,219

Trainable params: 348,219

Non-trainable params: 0
_________________________________________________________________
This also showed training loss: 0.0184 and validation loss: 0.0200. This wasn't any different than RELU activation. Also, I was able to train my model using this architecture but Model loading was hitting some error related to identifying Swish Activation function. Even after few google searches and solutions I couldn't resolve it and so moved on to other model design.

4) I had tried using RELU, ELU, Swish in Convolutional2D layers but couldn't get any different performance of Car on SImulator track. So I thought of altering few layers in my model. I removed 2 of the Convolutional2D layers and added 2 Dropout layers to see if there was any difference in training with these new architecture. I got training loss: 0.0202 and validation loss: 0.0228 . This was pretty much around the values received from my previous model and performance was little better. This model can be found as "model_red.h5" in the current directory. The simulator run on Track1 with this model can be found as run_red.mp4 in the same folder. The model summary is as below:

Layer (type)                 Output Shape              Param #   

=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 36)       2736      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 48)        43248     
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 77, 48)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 75, 64)        27712     
_________________________________________________________________
dropout_2 (Dropout)          (None, 12, 75, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 57600)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               5760100   
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        

=================================================================

Total params: 5,839,367

Trainable params: 5,839,367

Non-trainable params: 0
_________________________________________________________________

5) I was still not able to improve the performance on Track2 of simulator. So I tried to further enhance my model architecture by adding MaxPooling layer in it. I got training loss:0.0168 and validation loss: 0.0186. However the performance on Track2 with this model was better than previous models, though it couldn't still completely pass Track2. The run on Track1 of simulator was smoother and less jerky. So I decided to stay with this model. It can b found as "model.h5" in same folder and a video of its run on Track1 can be found as "run.mp4" in same folder. Its summary is as below:

Layer (type)                 Output Shape              Param #   

=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 36)       2736      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 15, 79, 36)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 38, 48)         43248     
_________________________________________________________________
dropout_1 (Dropout)          (None, 6, 38, 48)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 36, 64)         27712     
_________________________________________________________________
dropout_2 (Dropout)          (None, 4, 36, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               921700    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11    

=================================================================

Total params: 1,000,967

Trainable params: 1,000,967

Non-trainable params: 0
_________________________________________________________________


I have used Generator to build and train the model with 5 Epochs. I have used Adam optimiser so the learning rate was not tuned manually and Mean Square Error Loss function. I have saved the model after training as model.h5 and output Model Summary. These steps can be found at lines 167-175 of model.py.

