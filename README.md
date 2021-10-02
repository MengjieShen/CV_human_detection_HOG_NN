# Human Detection using HOG and NN
This is an implementation of Human Detection using Histogram Oriented Gradient and a two layer perceptron neural network for detecting human. 
The project consists of four steps:
• Greyscale color image 
• Gradient operation (Prewitt's Operator)
• Compute HOG features
• Backpropogation using a two-layer perceptron

There are 20 training images (10 positive and 10 negative) and 10 test images (5 positive and 5 negative) in .bmp format. 
All images are of size 160 (Height) X 96 (Width). Using the parameters specified in the project description, you should have 20 X 12 cells and 19 X 11 blocks. The size of your final HOG descriptor should be 7,524 X 1.

## How to compile and run my program
The only library functions that I use are numpy those for the reading and writing of image files, matrix and vector arithmetic, and certain other commonly used mathematical functions. You can directly run the project2.ipynb file. In the section of training neural network, you can specify the number of neurons and input.

## Functions

### Converting Color Image to Black and White 

Converted the color image into a greyscale image using the formula Img =(0.299𝑅 + 0.587𝐺 + 0.114𝐵) where R, G and B are the pixel values from the red, green and blue channels of the color image.

### Gradient Operation (Prewitt's Operator)
The Prewitts operator is used to compute x and y image gradients from the grayscale image and then compute edge magnitude and gradient angles.

### HOG Feature
I used the unsigned representation and quantized the computed gradient angle into one of the 9 bins. If the gradient angle is in the range [170, 350) degrees,I subtracted by 180 first. I used the following parameter values in my implementation: cell size = 8 x 8 pixels, block size = 16 x 16 pixels (or 2 x 2 cells), block overlap or step size = 8 pixels (or 1 cell.) I used L2 norm for block normalization. 

### Backpropogation using a Two-Layer Perceptron
I implemented a two-layer perceptron for classifying the images represented by their HOG descriptor into human or no-human. The perceptron have an input layer of size N, with N being the size of the HOG descriptor, a hidden layer and an output layer with one output neuron. I tried hidden layer sizes of 250, 500 and 1,000 neurons. I use the ReLU activation function for neurons in the hidden layer and the Sigmoid function for the output neuron. The Sigmoid function will ensure that the output is within the range [0,1], which can be interpreted as the probability of having detected human in the image.