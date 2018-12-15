# Fizzbuzz-using-Neural-Network
This code gives a contrast difference between the traditional programming methodology and Machine learning model. I have leveraged the Fizz-buzz problem to point out the intricacies and advantages ML (Software 2.0) proposes over conventional programming model (Software 1.0)


## Implementation
The Software 1.0 returns an hundred percent accuracy because the algorithm works for all test cases and therefore does not predict the supposed output. Rather it follows an imperatively valid formula to identify the result. The output of this algorithm is used as a standard for SOftware 2.0
Whereas in Software 2.0, the algorithm learns based on the test cases provided, and makes a decision with the training. Thus the results can be tweaked if the hyperparameters of the function is changed.





###  Generating the training dataset

We use software 1.0 to generate ground-truth values. A sample python script is written to loop from 101 to 1000. 
The ML model requires a label to recognize the result. So the 'Fizz/Buzz/Fizzbuzz/None' results are tabulated and are appended along with the number. 

###  Creating the model

We then move on to define the model with the input, output and hidden layers. 
The weights are defined and the learning rate is chosen by trial and error method
The activation function is chosen and the hidden \& output layer computation is defined. 
Also the loss functions and SGD functions are defined.

###  Training the model

The Epoch and the Batch size are decided and the model is trained with a value randomly generated from the training set. Based on the Epoch value, the total training set is looped through multiple times to help the model generate patterns.The model is then tested with the test set which consists of values from 1 to 100. 

###  Tweaking the model

The accuracy of the model is then compared with the obtained result and the original label of the test set. Based on the output, hyperparameters are changed to obtain better results. 
