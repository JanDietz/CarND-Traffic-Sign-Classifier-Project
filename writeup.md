# **Traffic Sign Recognition** 

## Writeup, Jan Dietz


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/processed.jpg "Original vs. Processed"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/web_signs.jpg "Traffic Signs from web"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the shape method to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed over the classes. Some Sign-Classes appears much more times than others.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it turned out better results with grayscale images. After that I applied the standard score method to normalize the data, because this helps to overcome the issue that pictures were made in different environments. Now the values are related to a mean of 0 and a deviation of 1. The following a traffic sign before and after image processing:

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16  |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	      	| Output = 400 				|
| Fully connected		| Output = 120  	|
| RELU					|												|
| Fully connected		| Output = 84  	|
| RELU					|												|
| Dropout					|		keep_prob= 0.5/1.0 	|
| Fully connected		| Output = 43  	|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

In order to train the model I used a batch size of 128, a learning rate of 0.001, and epoch count of 100.
I experimented with these parameters, but I got the best accuracy with batch size of 128, a learning rate of 0.001. I increased epoch to 100. I noticed that the test set accuracy still increases, even if the validation accuracy stucks or decreases from epoch 60 and higher.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.960
* test set accuracy of 0.944

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I started by using the default LeNet architecture. I changed it in order to accept grayscale images and to output the 43 different class types.

* What were some problems with the initial architecture?
LeNet architecture does not have any dropout. I added dropout to avoid overfitting on the training data.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Nothing more was changed.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet Architecture
* Why did you believe it would be relevant to the traffic sign application?
I did not change it much, because it already showed good results.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The test set was not used for training. Therefore test set accuracy of 0,944 is the evidence that this network is working. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image4]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No entry     			| Speed limit (30km/h) 					|
| Ahead only				| Ahead only										|
| Right-of-way at the next intersection	      		| Right-of-way at the next intersection						 				|
| Priority road			| Priority road      							|
| Speed limit (30km/h)			| Speed limit (30km/h)	     							|
| Yield			| Yield    							|
| Road work			| Road work    							|


The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. This is worse than the test set(94.4%). The No entry sign was predicted as Speed limit (30km/h). That reveals the problem of different occurences of classes in a data set. There are more examples of Speed limit (30km/h)(~2000 counts) than no entry signs (~1000 counts).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         	| Stop sign   									| 
| 0.0     				 | Speed limit (20km/h)										|
| 0.0					     | Speed limit (30km/h)										|
| 0.0	      			| Speed limit (50km/h)				 				|
| 0.0				      | Speed limit (60km/h)    							|


For the second image I was surprised why the no entry sign was ranked on 3 with just about 0.05%.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.729         	| Speed limit (30km/h)   									| 
| 0.269     				 | Speed limit (120km/h)									  |
| 0.0005					    | 	No entry									|
| 0.0000003      			| 	Keep right			 				|
| 0.0000003				      |  Speed limit (80km/h)   				|

Images 3-7 have the same probabilities as the first image. But for the last image we see that the probability is 99.96%. I did not expect such a good result for the Road work sign, because it is often poorly recorded. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9996         	| Road work  									| 
| 0.0004    				 | Speed limit (80km/h)								  |
| 0.000001					    | 	Road narrows on the right								|
| 0.0    			| 	Traffic signals		 				|
| 0.0		      |  Bicycles crossing 				|

As a conclusion the network learns those traffic signs very deep that are very common in the data set. As a improvement the data set classes could be balanced by copying examples from less common classes and adding to the dataset. That would increase the punishment for small classes in case of a misprediction.


