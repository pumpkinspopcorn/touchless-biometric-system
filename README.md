# touchless-biometric-system
In today’s era, usage of biometric system for identifying a person for their entry in a particular place or for an attendance system is getting common. In this project, we create an algorithm for real time face detection and reading ID cards to identify a person. Algorithm like Principal Component Analysis (PCA) is used for reduction of face space dimension and then used to obtain the image characteristics using Fisher Liner Discriminant (FLD) also known as Linear Discriminant Analysis (LDA). LBP (Local Binary Pattern) is another technique used too. We match the captured temporary data with the already existing data set that we will be using for system training. The system testing this will be basically done by process of feature vector and patter matching. For face recognition Haar feature-based Cascade Classifier (OpenCV) is used. It is a machine learning based approach where a cascade function is  trained from a lot of positive and negative images. It is then used to detect objects in other images. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. The method used in this study is the literature study that is studying and reviewing various books or literature related to mathematical concepts that underlies the formation of LBP (Local Binary Pattern) algorithm to recognize the image of a person's face.

## methodology

![image](https://github.com/user-attachments/assets/1d56ff66-3afa-4081-af13-f62353df18a4)

## libraries

Make sure to install all the libraries, if any library is missing it will prompt and you can install the one which has been prompted as missing. If an error is shown, make sure to update the library to the latest version. I used Anaconda for coding.

* OpenCV
* NLTK
* numpy
* keras_preprocessing
* tensorflow
* pycache
* pytesseract
* python imaging library

## techniques used

### for face recognition -

* Local Binary Pattern Algorithm (LBP)
* ADA Boost and Cascade Classifier ( very useful for gaining faster face recognition output
* Haar Cascade Classifier

### for text recognition from an ID card

* Optical Character Recognition (OCR) using pytesseract

## working model of face recognition

![image](https://github.com/user-attachments/assets/a04fbd99-894d-43ca-89fc-65b05f85ed66)


## working model of text recognition

![image](https://github.com/user-attachments/assets/49ae7053-cdb0-46b8-b281-606e32cb6179)

## Using LBP (Local Binary Pattern) algorithm over Fisher Face Method (PCA and LDA algorithm)

Fisher Face method is a technique used for face detection which uses both PCA (Principal Component Analysis) and LDA (Linear Discriminant Analysis) which calculates the eigen faces and fisher faces respectively. Here FisherFace will recognize the face by matchingit with the already received result of feature extraction. This is a combination of PCA and LDA methods. Where before performing LDA process the PCA is used to first solve singular problems by reducing the dimensions before being used to perform the LDA.

## So, which ones better for face recognition? Haar Cascade or LBP?

LBP was pretty fast, oh well !!

![image](https://github.com/user-attachments/assets/3f004bdf-1205-4c08-8db3-07818c695988)

## how to process the text recognition

You gotta make your own dataset using a notepad. You need to have a physical id with you and add all the details written in that id to the notepad. Run it and the OCR will match the details of the physical id with the database you created.
For example - 

![opencv_frame_0](https://github.com/user-attachments/assets/1d9f312b-678f-4293-983b-3738ff66f1b2)

## how to process the face recognition

The same goes for face recognition where you have to take sample images of yourself or the person you want to register to ( it would be a plus to take atleast 500 pictures of yourself looking left, right, straight, making expressions).
Once done, make a folder where you will store all the pictures and will be used in the code.






