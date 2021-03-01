# SureStart
A repository for the Spring 2021 AI & Machine Learning trainee program at [SureStart](https://mysurestart.com/), an organization connecting emerging AI talent to comprehensive technical curriculum and industry mentorship. 

<br>

<img src="https://static1.squarespace.com/static/5f45536caa356e6ab51588f4/t/5f57a9f14783ec36d7afc344/1608331568995/"/>

<br>

Program training includes machine learning, natural language processing, computer vision, data analytics/visualization, and ethical AI development. Training is followed by a hackathon, where teams build products with AI/ML technologies to solve real-world problems.

## Responses

### Day 19 (February 26th, 2021): Upsampling and Autoencoders

Today, I learned about upsampling and autoencoders and some use cases for getting an input image back from downsampled numbers. An autoencoder is a neural network that learns representations of data in an unsupervised manner. It consists of an encoder, which learns a representation of input data, and a decoder which decompresses data to reconstruct the input.

### *Day 18 (February 25th, 2021): Sentiment Analysis with Neural Networks and Regularization*

Code for neural network I built today can be found [here](https://github.com/natasharavinand/SureStart/tree/main/Sentiment%20Analysis%20With%20Neural%20Networks). 

Today, I was acquainted with various regularization techniques to decrease overfitting in a neural network (ex. reduced capacity, L1/L2 regularization, and dropout). I built a neural network inspired by [this](https://towardsdatascience.com/handling-overfitting-in-deep-learning-models-c760ee047c6e#:~:text=Overfitting%20occurs%20when%20you%20achieve,are%20irrelevant%20in%20other%20data.&text=The%20best%20option%20is%20to%20get%20more%20training%20data) tutorial that classified Tweets based on their sentiment and dealt with overfitting by demonstrating the above techniques.



### *Day 17 (February 24th, 2021): Predicting Home Prices and Loss Functions*

Code for neural network I built today can be found [here](https://github.com/natasharavinand/SureStart/tree/main/Predict%20House%20Price%20With%20CNNs). Today, I built a CNN model to accurately predict house prices.

Notes from [How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/).

**Deep learning neural networks**  are trained using the stochastic gradient descent optimization algorithm.

As part of the optimization algorithm, the error for the current state of the model must be estimated repeatedly. This requires the choice of an error function, conventionally called a  **loss function**, that can be used to estimate the loss of the model so that the weights can be updated to reduce the loss on the next evaluation.

Loss functions can often be divided into 3 categories: regression loss functions, binary classification loss functions, and multi-class classification loss functions.

### Regression Loss Functions

The **Mean Squared Error**, or MSE, loss is the default loss to use for regression problems. 

There may be regression problems in which the target value has a spread of values and when predicting a large value, you may not want to punish a model as heavily as mean squared error. Instead, you can first calculate the natural logarithm of each of the predicted values, then calculate the mean squared error. This is called the **Mean Squared Logarithmic Error** loss, or MSLE for short. 

On some regression problems, the distribution of the target variable may be mostly Gaussian, but may have outliers, e.g. large or small values far from the mean value. The **Mean Absolute Error**, or MAE, loss is an appropriate loss function in this case as it is more robust to outliers. It is calculated as the average of the absolute difference between the actual and predicted values.

### Binary Classification Loss Functions

Binary classification are those predictive modeling problems where examples are assigned one of two labels.

**Cross-entropy** is the default loss function to use for binary classification problems. It is intended for use with binary classification where the target values are in the set {0, 1}.

An alternative to cross-entropy for binary classification problems is the  **hinge loss function**, primarily developed for use with Support Vector Machine (SVM) models. It is intended for use with binary classification where the target values are in the set {-1, 1}. A popular extension is called the **squared hinge loss** that simply calculates the square of the score hinge loss.

### Multi-Class Classification Loss Functions

Multi-Class classification are those predictive modeling problems where examples are assigned one of more than two classes.

**Cross entropy** is the default loss function to use for multi-class classification problems. In this case, it is intended for use with multi-class classification where the target values are in the set {0, 1, 3, …, n}, where each class is assigned a unique integer value. This will require a one-hot encoding process.

A possible cause of frustration when using cross-entropy with classification problems with a large number of labels is the one hot encoding process. **Sparse cross-entropy** addresses this by performing the same cross-entropy calculation of error, without requiring that the target variable be one hot encoded prior to training.

<hr>

### *Day 16 (February 23th, 2021): Activation Functions*

Notes from [How to Choose an Activation Function for Deep Learning](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/).

Activation functions are crucially important when designing a neural network architecture. A particular choice for an activation function in a hidden layer can control how well the model learns the training dataset, and a particular choice for an activation function in the output layer will control the kinds of predictions a model makes.

A network may have three types of layers: input layers that take raw input from the domain,  **hidden layers**  that take input from another layer and pass output to another layer, and  **output layers**  that make a prediction. Most hidden layers will typically use the same activation function, while the output layer will typically have a different activation function and will depend on the goal of the prediction of the model.

Activation functions are typically differentiable, which means the first-order derivative can be calculated for a given input value. This is crucial for backpropagation, the process in which the error of the model is minimized by updating weights of the model. 

### Activation Functions for Hidden Layers

The ReLU activation function is the most common function for hidden layers. It is common because it is simple to implement and overcomes the limitations of other popular activation functions such as sigmoid and tanh. Specifically, it tends to avoid the vanishing gradient problem,  where a deep multilayer feed-forward network or a recurrent neural network is unable to propagate useful gradient information from the output end of the model back to the layers near the input end of the model.

A graph of the ReLU activation function:

![ReLU activation function](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2020/12/Plot-of-Inputs-vs-Outputs-for-the-ReLU-Activation-Function..png)

Other popular activation functions include the sigmoid as well as the tanh.

Although ReLU is the most typically used activation function, your choice may depend on the type of neural network you're trying to build:

-   **Multilayer Perceptron (MLP)**: ReLU activation function.
-   **Convolutional Neural Network (CNN)**: ReLU activation function.
-   **Recurrent Neural Network**: Tanh and/or Sigmoid activation function.

### Activation Functions for Output Layers

The output layer is the layer in a neural network model that directly outputs a prediction. The three most common activation functions for output layers are linear, logistic (sigmoid), and softmax.

You must choose the activation function for your output layer based on the type of prediction problem that you are solving. For example, you may divide prediction problems into two main groups, predicting a categorical variable (_classification_) and predicting a numerical variable (_regression_).

-   **Regression**: One node, linear activation.
-    **Binary Classification**: One node, sigmoid activation.
-   **Multiclass Classification**: One node per class, softmax activation.
-   **Multilabel Classification**: One node per class, sigmoid activation.

![enter image description here](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2020/12/How-to-Choose-an-Output-Layer-Activation-Function.png)
 
<hr>

### *Day 15 (February 22th, 2021): Ethics Driven ML Practice*

Code for today can be found [here](https://github.com/natasharavinand/SureStart/tree/main/Classify%20Facial%20Data%20With%20CNNs).

Today, I learned about ethics-driven ML practice and how important it is to consider marginalized groups when conducting machine learning. I built a CNN that classified men and women, and noticed when reviewing my results that my model tended to incorrectly classify people of color a disproportionate amount. This reinforced my belief that staying vigilant about ethical software is crucial.

<hr>

### *Day 12 (February 19th, 2021): Image Classification with CNNs*

Code for today can be found [here](https://github.com/natasharavinand/SureStart/tree/main/Classify%20Animals%20with%20CNNs).

Today, I used a previously built CNN model to build a classifier that classifies cats vs dogs.

<hr>

### *Day 11 (February 18th, 2021): Classifying MNIST Digits with CNNs*

Code for today can be found [here](https://github.com/natasharavinand/SureStart/tree/main/Classify%20Digits%20With%20CNNs).

Today, I primarily worked to improve my previous CNN model that classified MNIST digits.

I also clarified my understanding of model depth and width. Model depth can be interpreted as the number of hidden layers in a neural network. It can often improve model accuracy, but we have to be careful not to overfit.

Model width can be interpreted as the number of nodes in each hidden layer. Although overfitting doesn't tend to be an issue with model width, we do have to be aware of training time, as a greater model width can lead to greater training times.

<hr>

### *Day 10 (February 17th, 2021): Neural Network Layers*

A convolutional neural network (CNN) is a neural network that is often used in computer vision or image recognition tasks. There are three main parts to a CNN:

- A convolution mechanism that uses filters to convolve around an image to distinguish features
- A pooling mechanism to reduce the dimensionality of the feature map without losing important information
- A fully connected layer that flattens the output of the above and classifies the image

Fully connected neural networks are different from fully connected layers in a CNN. In fully connected neural networks, all neurons connect to all neurons in the next layer. However, this architecture is inefficient for computer vision tasks, as images have an extremely large input (can have thousands of pixels). 

In contrast, a convolutional neural networks creates a mechanism for analyzing each feature in an image in isolation, rather than all the pixel data. In the convolutional neural network, the fully conneted layer is the "last step" that takes the result from the convolutional and pooling process to reach a classification decision.

<hr>

### *Day 9 (February 16th, 2021): Algorithmic Bias*

Today focused on making participants cognizant of algorithmic bias in some AI-based software, the reasons for its occurrence, and its effects on real-world inequalities and biases. In addition, the programming was set up to help participants think critically about the dangers that society might face, if automated systems have biases.

In the [Survival of the Best Fit](https://www.survivalofthebestfit.com/) interactive simulation, the player is walked through an example of how ML algorithms can perpetuate hiring biases. The simulation outlines the fact that ML models need large datasets that often include human biases in them. Thus, when the model is trained and deployed, it expressly learns those same biases. 

One notable example of how AI bias can have real-world consequences is the COMPAS algorithm. The COMPAS algorithm was developed to predict the tendency of a convicted criminal to reoffend. The results for the algorithm portrayed black and white offenders differently, with white offenders more likely to be labeled as lower risk and displaying lower probabilities to reoffend. However, when news organizations compared the COMPAS results with the actual data, they found that this wasn't true. In developing ML algorithms, we must consider societal factors to make sure we are not developing biased systems that cause systemic harm.

<hr>

### *Day 8 (February 15th, 2021): Introduction to Convolutional Neural Networks*

Code for the convolutional neural network built today is [here](https://github.com/natasharavinand/SureStart/tree/main/Classify%20Digits%20With%20CNNs).

Today, I learned more about convolutional neural networks and how they can classify images. I used the famous MNIST dataset to construct a neural network to classify handwritten digits with an accuracy of about 98.36% and a loss of 0.0549.

<hr>

### *Day 5 (February 12th, 2021): Introduction to Neural Networks*

Code for the convolutional neural network built today is [here](https://github.com/natasharavinand/SureStart/tree/main/Introduction%20to%20Neural%20Networks).

Today, we explored different components of neural networks, including concepts such as bias, activation functions, weights, densely connected layers, and epochs/batch-size. We explored how to create a convolutional neural network (CNN) that would use NLP techniques to classify news headlines from a Kaggle dataset as sarcastic or not sarcastic. 

<hr>

### *Day 4 (February 11th, 2021): Introduction to Deep Learning*

Deep learning is a subset of machine learning that uses deep learning algorithms to come up with important conclusions based on input data. Deep learning is usually unsupervised or semi-supervised. Some architectures of deep learning include convolutional neural networks, recurrent neural networks, generative adversarial networks, and recursive neural networks.

[This article](https://serokell.io/blog/deep-learning-and-neural-network-guide#what-are-artificial-neural-networks?) was provided as a primer for deep learning basics.

Deep learning can be used in a variety of fields, including NLP/sentiment analysis. Many modern social media platforms, including Twitter, struggle with monitoring whether content posted on the platform is safe or protected speech. NLP can help in making decisions that cultivate a healthier online ecosystem. One of the most popular datasets for NLP/sentiment analysis is [Sentiment140](http://help.sentiment140.com/for-students/). One could apply deep learning algorithms to be able to classify different kinds of Tweets from the vast array of Twitter data generated by billions of users. If I were to develop a model to do this, I would use recurrent neural networks (RNNs) as they are conventionally used in any sort of application regarding language.

<hr>

### *Day 3 (February 10th, 2021): Introduction to Tensorflow*

Today's materials introduced trainees to a high-level understanding of TensorFlow. The following two questions were posed:

- **What are "tensors" and how are they used in machine learning?**
	- A tensor is a mathematical construct that allows us to describe physical quantities. Tensors can have various ranks, and are often represented using n-dimensional arrays. In practice, tensors can be considered as dynamically-sized multidimensional data arrays.
	- To understand tensors, we must understand vectors well. Vectors can represent quantities with magnitude and direction which are a part of larger category called tensors. In order to understand this fully, we must understand vector components and basis vectors. A coordinate system (such as the Cartesian coordinate system) comes along with coordinate basis vectors, such as unit vectors. In a graphical representation, we can find different vector components by projecting onto axes.
	- *Scalar* = Tensor of Rank 0, magnitude and no direction, 1 component
	- *Vector* = Tensor of Rank 1, magnitude and direction, 3 components
	- *Dyad* (matrix) = Tensor of Rank 2, magnitude and direction in x, y, and z, 27 components (3 x 3)
	- A very high-quality introductory video for tensors can be found [here](https://www.youtube.com/watch?v=f5liqUk0ZTw&ab_channel=DanFleisch).
- **What did you notice about the computations that you ran in the TensorFlow programs (i.e. interactive models) in the tutorial?**
	- In TensorFlow, developers are able to create graphs (series of processing nodes). Each node in the graph represents a mathematical operation, and each connection or edge between nodes is a multidimensional data array, or a tensor. TensorFlow supports dataflow programming; dataflow programming models a program as a directed graph of the data flowing between operations.
	- In Tensorflow, developers work to build a computational graph and then execute the graph in order to run the operations.

<hr>

### *Day 2 (February 9th, 2021): Introduction to Machine Learning and scikit-learn*

Code and materials for today can be found [here](https://github.com/natasharavinand/SureStart/tree/main/Introduction%20to%20ML%20and%20scikit-learn).

Today introduced machine learning theory and applications, as well as model development in `scikit-learn`. The two discussion questions included:

 - **What is the difference between supervised and unsupervised learning?**
	 - The ultimate goal of many supervised learning problems is to come up with a predictor function (sometimes called hypothesis)  `h(x)`. Mathematical algorithms optimize this function such that given input data  `x`  (ex. square footage of a house), the predictor function can accurately output some value  `h(x)`  (ex. market price of house).
	 - The goal of unsupervised learning is to uncover patterns and relationships within data. In this case, there are no training samples. Some popular approaches include clustering (ex. k-means), dimensionality reduction (ex. principle component analysis) and more.
- **Does `scikit-learn` have the power to visualize data by itself?**
	- No. `scikit-learn` must be complimented by another library such as `Matplotlib` in order to visualize data.

<hr>

### *Day 1 (February 8th, 2021): Welcome to SureStart!*
Today was an introduction to the Spring 2021 SureStart program, an initiative connecting emerging AI talent to machine and deep learning curriculum. I am an undergraduate student at Yale University planning a tentative major in Statistics & Data Science. I'm interested in how to strategize, develop, and apply AI techniques to solve problems. I hope to strengthen my skills in AI/ML within the scikit-learn, Tensorflow, and Keras libraries. More broadly, I hope to develop skills in:

 - Unsupervised and supervised learning techniques
 - Deep learning and neural networks
 - NLP and computer vision
- Ethical AI development

I hope to eventually apply my technical skills in roles that require an understanding of AI/ML techniques to make data-driven decisions. Some fields that especially interest me are fintech and health tech.
