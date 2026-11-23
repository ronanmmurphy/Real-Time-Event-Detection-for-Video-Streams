# Real-Time-Event-Detection-for-Video-Streams
Designed and implemented Computer Vision pipeline for Video Stream object detection and classification. This Real-Time classifier used YOLO, MobileNet, and OpenCV to detect cars and classify their type and colour. The evaluation of the video was compared with a ground truth file which our pipeline which we calculated F1-Scores. The system scored 94%, 85%, and 90% for the detection of cars, classification of car type, and classification of car colour respectively. 


The pipeline was built using Python, due mainly to the fact that Python contains many libraries which provide classes, methods and other functionality for working with Deep Neural Networks (Tensorflow and Keras) and Colour Filtering and Detection (OpenCV). The pipeline is split into four stages, these are: Pre-processing, YOLO Object Detection, Car Type Classifier and Colour Classifier. Each step is described in detail below. Each stage is described in detail below.

![pipeline](https://github.com/ronanmmurphy/Real-Time-Event-Detection-for-Video-Streams/blob/main/Images/pipeline.PNG?raw=true)

Pipeline Input and Preprocessing: Stream the video.mp4 stream into the system by using a
video reader at a rate of 30 frames per second. This requires them to read the video
file and simulate a stream of image frames as individual data items.
 
 
Model Cascade: The model cascade will be a 3-stage computer vision pipeline. The
following steps need to be performed to build the model cascade:

● Stage 1: Object Detector: Deployed a state-of-the-art object detector model (TinyYolo),
pre-trained on MSCOCO dataset. 
A TinyYOLO model was implemented using an in-built MSCOCO dataset, which can classify up to 80 predefined objects.The TinyYOLO model weights and configuration file were combined to output the model as a HDF5, which defines the layers in the CNN (Convoluted Neural Network) assigning the initial weights to nodes. The required dataset of image frames created from the raw video are run through the model to detect the car objects in each frame. This would mean iterating through these images as a stream instead of calling the files individually.

The ‘detect_image’ method in the ‘yolo_video.py’ file was changed to open all the images in the folder and detect the objects outputting an image with rectanglular boundaries over objects found. The dimensions of this rectangle were also outputted so the images could be cropped for future stages. The time for each frame to return is also outputted for analysis and the total count of cars in the frame. The image returned is a jpeg file but as an object of the Pillow image library, which can display and crop the image when needed.
Another issue we encountered was that the MSCOCO dataset detects objects other than cars, such as people and bicycles, which aren’t needed for this implementation. This needed to be configured to allow only car object detection or the accuracy of the detector would suffer. This was edited by creating a filter in the yolo.py file to only add the dimensions of the objects which are of class “car” to a list which is returned for each frame. Once these changes were implemented the model could iterate through all the images and output to stage 2 the information and the frame required to classify the car type.

● Stage2- Attribute Classifier (Car Type): Trained Mobilenet Keras Library which is an existing pretrained object classifier then implemented a transfer learning approach for two car classes:

○ Sedan

○ Hatchback

The Classifier Model was trained in the following steps:
Step 1
In order to obtain car images of type Sedan and Hatchback it was necessary to create an image scraper to scrape these images from the internet. We adapted an image scraping example found onlinevii and
used this to scrape car images in bulk. We used 4 different search terms in 4 different scrapes in order to have as much variety in the training data as possible. It was necessary to have as many car images as possible as this would help to improve the accuracy of the model.
Step 2
After the images were scraped, the images were filtered manually in order to remove poor quality images. A poor quality image being a scraped image that was either not an image of a sedan or hatchback, or an image of a sedan or hatchback which would not aid our classifier in determining the difference between the two car type, e.g. a frontal image of a car which makes it impossible for a even a human to determine the type of car. After filtering the images, we obtained approximately 930 images which we used in our training and testing of the model.
Step 3
The filtered images were split into training and testing subsets, with 80% of the images being used for training and 20% being used for testing. Furthermore, the training data was split into training and validation data, with 80% being used for training and 20% for validation. Two folders were created in the directory where the code was run from, one for training data and another for testing data, inside each folder were two subfolders named “Sedan” and “Hatchback”. Inside both of these folders were jpeg images of those car types, the name of the folder being the ‘label’ of the images.
Step 4
To train our car classifier model, we adapted some examples of CNN architectures found onlineviii. We applied transfer learning by retraining the existing Mobilenet object classifier, only retraining the top 3 layers of this original CNN. The reason the model is trained in this way using Transfer Learning is that we can “extract useful information from data in a related domain”ix, in this case the Mobilenet pretrained classifier, and “transfer (it) for being used in target tasks”x, i.e. to classify car types. We inserted a dropout layer, a dense (fully connected) layer and an activation layer to classify car type in that order as the final 3 layers of the model. The model was trained for 70 epochs and a graph of the training and validation loss for each epoch is plotted below.


![Model Loss](https://github.com/ronanmmurphy/Real-Time-Event-Detection-for-Video-Streams/blob/main/Images/ModelLoss.PNG?raw=true)

● Stage3- Attribute Classifier (Car Color): - Createc an attribute classifier for five color
classes using:

○ Red

○ Black

○ White

○ Blue

○ Silver

Each input image is first scaled in order to make each image of equivalent size, this is also done as if the images are scaled down in size there are less colour pixels to check in the image, which reduces the time taken to detect a colour. Following this the images are converted to a HSV colouring from their original RGB colouring, this type of colouring was chosen as it is the method most like the way in which humans perceive colour. Five HSV colour ranges were created using the Colourizer websitev, these ranges were for the colours black, white, silver, red and green. Inputting HSV colour values in OpenCV is not straightforward. HSV colour values are not represented in the conventional way in OpenCV where the value of H (Hue) lies in the range 0-360, S (Saturation) lies in the range 0-100 and V (Value) also lies in the range 0-100. As a result, a short python script to convert conventional HSV values which could be read from Colourizer to OpenCV suitable HSV values was createdvi. Having defined the lower and upper HSV ranges for each colour, the inRange() OpenCV function is used to detect pixels in an image within a given colour range. This function returns a value of 255 for pixels which are a colour within the specified range and a value of 0 for pixels that are not a colour in the given range. Using the inRange() function the image is tested on each colour range using a loop. The overall detected colour of the image was the colour range which produced the highest number of pixels with value 255.

Evaluation:
The evaluation of the pipeline is done by comparing the output of each of the three queries to a pre-defined ‘ground-truth’ dataset, which has the statistics for counts of each category at each frame. The values for True Positives, True Negatives, False Positives and False Negatives for each query are calculated for each frame and the precision and recall of each query is calculated. Following this an overall F1_Score for each query was calculated and is given below.

![F1-scores](https://github.com/ronanmmurphy/Real-Time-Event-Detection-for-Video-Streams/blob/main/Images/f1scores.PNG?raw=true)
