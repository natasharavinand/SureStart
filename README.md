# SureStart
A repository for the Spring 2021 AI & Machine Learning trainee program at [SureStart](https://mysurestart.com/), an organization connecting emerging AI talent to comprehensive technical curriculum and industry mentorship. 

<br>

<img src="https://static1.squarespace.com/static/5f45536caa356e6ab51588f4/t/5f57a9f14783ec36d7afc344/1608331568995/"/>

<br>

Program training includes machine learning, natural language processing, computer vision, data analytics/visualization, and ethical AI development. Training is followed by a "hackathon", where teams build products with AI/ML technologies to solve problems in affective computing.

## Responses

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
