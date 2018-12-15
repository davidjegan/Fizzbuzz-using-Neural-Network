#Import the headers required 
#1. Numpy - used to store the input/output array
#2. Tensorflow - ML framework that offers predefined methods 
#3. TQDM - To print the output for processing (epochs) in Notebook
#4. Pandas - A data manipulation framework which can store large data as dataframes
#5. Keras - Higher level wrapper for tensorflow. 
#6. Matplotlib - A visualizing tool

import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook
import pandas as pd
from keras.utils import np_utils
import matplotlib.pyplot as plt
#%matplotlib inline


def createInputCSV(start,end,filename):
    
    # List can hold data in an array and it can be mutable
    inputData   = []
    outputData  = []
    
    # We need to assign label to each input value. So we append it in each line
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))
    
    # Dataframe is a dictionary type which contains key,value relation
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)
    
    print(filename, "Created!")

#Sotware 1.0
def fizzbuzz(n):
    
    # If the number is divisible by 3, 5, 15, we print Fizz, Buzz or Fizzbuzz respectively. Else we print the number as is. 
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'

#Gathering training data
#1. processData - Input requires both number and label for the model to understand what exactly the pattern refer to
#2. encodeData  - We convert the input to an array of activations (int to Binary) to gather more correlation between the input and label
#3. encodeLabel - This has the actual definition for Labelling the inputs 

def processData(dataset): 
    #We get the input and invoke the encodeData and encodeLabel methods for extracting labels from the dataset
    #We process the data because, initially it would be an integer. But the model needs more information to identify a pattern
    data   = dataset['input'].values
    labels = dataset['label'].values
    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)
    return processedData, processedLabel

def encodeData(data):
    #The input is expanded from the int to binary. 
    #The maximum number here is 4000, which is 2 power 12. So we consider 12 bits to expand number from 1 value to 12 vales. 
    processedData = []
    for dataInstance in data:
        processedData.append([dataInstance >> d & 1 for d in range(12)])
    return np.array(processedData)


def encodeLabel(labels):
    #based on the labelinstance we assign values to the processedLabel list
    processedLabel = []
    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])
    return np_utils.to_categorical(np.array(processedLabel),4)


# We create test, training sets
createInputCSV(101,4001,'training.csv')
createInputCSV(1,101,'testing.csv')


# Read Dataset
trainingData = pd.read_csv('training.csv')
testingData  = pd.read_csv('testing.csv')

# Process Dataset
processedTrainingData, processedTrainingLabel = processData(trainingData)
processedTestingData, processedTestingLabel   = processData(testingData)


# Placeholders are used to build the computational graph thuogh the data is not available at the moment
# The argument None denotes, the number of results is not provided, but the size is fixed (12 or 4)
inputTensor  = tf.placeholder(tf.float32, [None, 12])
outputTensor = tf.placeholder(tf.float32, [None, 4])

#The neuron count and the learning rate is decided
NUM_HIDDEN_NEURONS_LAYER_1 = 100
LEARNING_RATE = 0.15


#The Python array is passed as the shape argument to method tf.random_normal 
#The weights are decided randomly using tf.random_normal 
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.02))

# Initializing the input to hidden layer weights
input_hidden_weights  = init_weights([12, NUM_HIDDEN_NEURONS_LAYER_1])
# Initializing the hidden to output layer weights
hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 4])

# Computing values at the hidden layer
# The activation function is chosen  
# Computing values at the output layer


#The matrix multiplication of the inputTensor and input_hidden_weights  
#This is a tensor of size [sample count * NUM_HIDDEN_NEURONS_LAYER_1]
hidden_layer = tf.nn.relu(tf.matmul(inputTensor, input_hidden_weights))

#The matrix multiplication of the hiddenlayer and input_output_weights  
#This is a tensor of size [sample count * 4]
output_layer = tf.matmul(hidden_layer, hidden_output_weights)


#Defining Error Function
#To measure the loss, we use cross entropy
#softmax is used as a normalization technique 
error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))

# Defining Learning Algorithm and Training Parameters
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)

# Prediction Function. We print out the largest value in the tensor
prediction = tf.argmax(output_layer, 1)


#No of epoch is the number of times we completely go through the input set
#Batch size denotes the processing size considered

NUM_OF_EPOCHS = 1500
BATCH_SIZE = 200

training_accuracy = []

#We initialize the Tensorflow session
with tf.Session() as sess:
    
    # Set Global Variables because this initializes all the variables 
    tf.global_variables_initializer().run()
    
    for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):
        
        #Shuffle the Training Dataset at each epoch
        p = np.random.permutation(range(len(processedTrainingData)))
        #We train the model with the random 'p' and get its label and binary value
        processedTrainingData  = processedTrainingData[p]
        processedTrainingLabel = processedTrainingLabel[p]
        
        # Start batch training
        for start in range(0, len(processedTrainingData), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(training, feed_dict={inputTensor: processedTrainingData[start:end], 
                                          outputTensor: processedTrainingLabel[start:end]})
        # Training accuracy for an epoch
        # The np.argmax returns a tensor of values in 0,1,2 and 3. The mean returns the average of accuate prediction and this is appended to the list training_accuracy 
        training_accuracy.append(np.mean(np.argmax(processedTrainingLabel, axis=1) ==
                             sess.run(prediction, feed_dict={inputTensor: processedTrainingData,
                                                             outputTensor: processedTrainingLabel})))
        
    # Testing
    predictedTestLabel = sess.run(prediction, feed_dict={inputTensor: processedTestingData})


def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"
    
wrong   = 0
right   = 0
f  = 0
b = 0
fb = 0
o = 0

predictedTestLabelList = []
""
for i,j in zip(processedTestingLabel,predictedTestLabel):
    predictedTestLabelList.append(decodeLabel(j))
    
    if np.argmax(i) == j:
        right = right + 1
        if(j == 1): 
            f = f + 1
        if(j == 2): 
            b = b + 1
        if(j == 3): 
            fb = fb + 1
        if(j == 0): 
            o = o + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))
print("Accuracte Fizz     - "  + str(f) )
print("Accuracte Buzz     - "  + str(b) )
print("Accuracte Fizzbuzz - "  + str(fb) )
print("Accuracte Others   - "  + str(o) )


# Please input your UBID and personNumber 
testDataInput = testingData['input'].tolist()
testDataLabel = testingData['label'].tolist()


predictedTestLabelList.insert(0, "")
predictedTestLabelList.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabelList

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')