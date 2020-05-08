#************************************************************************************************************************************************************#
#                                                                                                                                                           *#
#    Project Title    : Deep Neural Network for Cat Calssifier                                                                                              *#
#    Description      : A deep neural network for image classification , with a Configurable number of layers                                               *#
#                                                      # MODEL ARCH #                                                                                       *#
#                       The model can be summarized as: [LINEAR -> RELU]  ××  (L-1) -> LINEAR -> SIGMOID                                                    *#
#                       The input is a (64,64,3) image which is flattened to a vector of size (12288,1).                                                    *#
#                       The corresponding vector:  [x0,x1,...,x12287]T[x0,x1,...,x12287]T  is then multiplied by the weight matrix  W[1]W[1]  and then      *#
#                       you add the intercept  b[1]b[1] . The result is called the linear unit.                                                             *#
#                       Next, you take the relu of the linear unit. This process could be repeated several times for each                                   *#
#                       (W[l],b[l])(W[l],b[l])  depending on the model architecture.                                                                        *#
#                       Finally, you take the sigmoid of the final linear unit. If it is greater than 0.5, you classify it to be a cat.                     *#
#                       3.3 - General methodology                                                                                                           *#
#                       As usual you will follow the Deep Learning methodology to build the model:                                                          *#
#                       1. Initialize parameters / Define hyperparameters                                                                                   *#
#                       2. Loop for num_iterations:                                                                                                         *#
#                         a. Forward propagation                                                                                                            *#
#                         b. Compute cost function                                                                                                          *# 
#                         c. Backward propagation                                                                                                           *#
#                         d. Update parameters (using parameters, and grads from backprop)                                                                  *#
#                       3. Use trained parameters to predict labels                                                                                         *#
#                       Notations : Superscript  [l][l]  denotes a quantity associated with the  lthlth  layer.                                             *#
#                       Example:  a[L]a[L]  is the  LthLth  layer activation.  W[L]W[L]  and  b[L]b[L]  are the  LthLth  layer parameters.                  *#
#                       Superscript  (i)(i)  denotes a quantity associated with the  ithith  example.                                                       *#
#                       Example:  x(i)x(i)  is the  ithith  training example.                                                                               *#
#                       Lowerscript  ii  denotes the  ithith  entry of a vector.                                                                            *#
#                       Example:  a[l]iai[l]  denotes the  ithith  entry of the  lthlth  layer's activations).                                              *#
#    File             : Main File                                                                                                                           *#
#    Date             : 29/4/2020                                                                                                                           *#
#    Reference        : NONE                                                                                                                                *#
#                                                                                                                                                           *#
#************************************************************************************************************************************************************#
#   Author  : Yousef Hesham                                                                                                                                 *#
#************************************************************************************************************************************************************#   

import time
import numpy as np #is the fundamental package for scientific computing with Python.
import h5py #is a common package to interact with a dataset that is stored on an H5 file.
import matplotlib.pyplot as plt #matplotlib is a library to plot graphs in Python.
import scipy
from PIL import Image
from scipy import ndimage
from helper_functions import *
from math_functions import *


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# FUNCTION: DISPLAY NUMBERS AND DATA
def Display():

    # Example of a picture
    index = 208 #Choose from 209 Dataset samples, from index 0 to 208 
    plt.imshow(train_x_orig[index])
    print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
    plt.show()

    # Explore your dataset 
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))


# FUNCTION: Loading DATA SET
def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# FUNCTION: Predict the results of A L-layer NN
def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


#### MAIN ####

# Loading Data set
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
num_px = train_x_orig.shape[1]

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

# CONSTANTS 
layers_dims = [12288, 20, 7, 5, 1] # INPUT OF DIMS: 12288 AND 4-Hidden layer model


# IMPLIMETATION
parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.0075 ,num_iterations = 2500, print_cost = True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)


#THIS SECTION IS TO TEST YOUR OWN IMAGE / Replace photo.jpeg with you image to test
my_image = "cat.jpeg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "testimage/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
plt.title("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
plt.show()
