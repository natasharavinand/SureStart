# SureStart
A repository for the Spring 2021 AI & Machine Learning trainee program at [SureStart](https://mysurestart.com/), an organization connecting emerging AI talent to comprehensive technical curriculum and industry mentorship. 

<br>

<img src="https://static1.squarespace.com/static/5f45536caa356e6ab51588f4/t/5f57a9f14783ec36d7afc344/1608331568995/"/>

<br>

Program training includes machine learning, natural language processing, computer vision, data analytics/visualization, and ethical AI development. Training is followed by a hackathon, where teams build products with AI/ML technologies to solve real-world problems.

## Responses

### *Day 8 (February 17th, 2021): Neural Network Layers*

A convolutional neural network (CNN) is a neural network that is often used in computer vision or image recognition tasks. There are three main parts to a CNN:

- A convolution mechanism that uses filters to convolve around an image to distinguish features
- A pooling mechanism to reduce the dimensionality of the feature map without losing important information
- A fully connected layer that flattens the output of the above and classifies the image

Fully connected neural networks are different from fully connected layers in a CNN. In fully connected neural networks, all neurons connect to all neurons in the next layer. However, this architecture is inefficient for computer vision tasks, as images have an extremely large input (can have thousands of pixels). 

In contrast, a convolutional neural networks creates a mechanism for analyzing each feature in an image in isolation, rather than all the pixel data. In the convolutional neural network, the fully conneted layer is the "last step" that takes the result from the convolutional and pooling process to reach a classification decision.

### *Day 7 (February 16th, 2021): Algorithmic Bias*

Today focused on making participants cognizant of algorithmic bias in some AI-based software, the reasons for its occurrence, and its effects on real-world inequalities and biases. In addition, the programming was set up to help participants think critically about the dangers that society might face, if automated systems have biases.

In the [Survival of the Best Fit](https://www.survivalofthebestfit.com/) interactive simulation, the player is walked through an example of how ML algorithms can perpetuate hiring biases. The simulation outlines the fact that ML models need large datasets that often include human biases in them. Thus, when the model is trained and deployed, it expressly learns those same biases. 

One notable example of how AI bias can have real-world consequences is the COMPAS algorithm. The COMPAS algorithm was developed to predict the tendency of a convicted criminal to reoffend. The results for the algorithm portrayed black and white offenders differently, with white offenders more likely to be labeled as lower risk and displaying lower probabilities to reoffend. However, when news organizations compared the COMPAS results with the actual data, they found that this wasn't true. In developing ML algorithms, we must consider societal factors to make sure we are not developing biased systems that cause systemic harm.

<hr>

### *Day 6 (February 15th, 2021): Introduction to Convolutional Neural Networks*

Code for the convolutional neural network built today is [here](https://github.com/natasharavinand/SureStart/tree/main/Introduction%20to%20CNNs).

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

I hope to eventually apply my technical skills in roles that require an understanding of AI/ML techniques to make data-driven decisions. Some fields that especially interest me are civic/progressive tech, climate tech, and fintech.
