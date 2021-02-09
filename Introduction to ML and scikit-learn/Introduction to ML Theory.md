---
output:
  pdf_document: default
  html_document: default
---
# Introduction to Machine Learning Theory and Applications

This are my notes regarding Machine Learning Theory and its applications, mostly drawn from the fantastic introductory article by Nicke McCrea on TopTal (link [here](https://www.toptal.com/machine-learning/machine-learning-theory-an-introductory-primer)). Josh Starmer of StatQuest also has a phenomenal video on the gradient descent algorithm (link [here](https://youtu.be/sDv4f4s2SB8)). Lastly, Grant Sanderson of 3Blue1Brown has an introductory and very popular video on neural networks [here](https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown).

Most of the information covered below touches on supervised learning algorithms.

## Overview

Machine learning (ML) is a rapidly expanding and developing field that encompasses numerous kinds of sub-disciplines. ML is already used in a wide array of industries and fields to condense big data into actionable insights. Though there is no official definition, Tom Mitchell of Carnegie Mellon University has stated a concisely of what ML could be considered to be:

> “A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.” 
*> Tom Mitchell, Carnegie Mellon University*

There are two main kinds of machine learning – supervised and unsupervised learning.

 - **Supervised Learning**: A computer program uses a training set of prelabeled data in order to inform its conclusions
 - **Unsupervised learning**: A computer program must uncover patterns in the data by itself

## Supervised Learning

The ultimate goal of many supervised learning problems is to come up with a predictor function (sometimes called hypothesis) `h(x)`. Mathematical algorithms optimize this function such that given input data `x` (ex. square footage of a house), the predictor function can accurately output some value `h(x)` (ex. market price of house).

`x` will almost always require multiple data points. For example, if one wanted to predict housing price and develop an appropriate predictor function, one might want to take into consideration:

 - `x1`: Square footage of house
 - `x2`: Number of bedrooms
 - `x3`: Number of floors
 - `x4`: Zip code
 - and more

A simple predictor function can have the form:
<br>

![h of x equals theta 0 plus theta 1 times x](https://uploads.toptal.io/blog/image/444/toptal-blog-image-1407508963294.png)
<br>
Where  ![theta 0](https://uploads.toptal.io/blog/image/445/toptal-blog-image-1407508999266.png) and ![theta 1](https://uploads.toptal.io/blog/image/446/toptal-blog-image-1407509052417.png) are constants. Our main goal in developing a predictor function is to find the "perfect" values of both such that our predictions are as close to reality as possible.

This optimization of `h(x)` requires training examples. By using a training set, we known in advance values of `y`. So, for each training example, we have an input value `x_train` for which its corresponding `y` is already known. Using this information, we can  find the difference between the known, correct value of `y` and our predicted value of `h(x_train)`. Then, we can try to minimize this "loss". In the end our predictor becomes trained and is able to start predicting on unknown values.

Real world data often has noise and is not very neat, which can make it difficult, if impossible, to make perfect guesses. However, as British Professor of Statistics George E. P. Box Once said:

> "All models are wrong, but some are useful."
*> Professor George E. P. Box*

We want to focus on building good models that can give good predictions, not necessarily perfect ones.

ML and statistics are interrelated, and thus important concepts in statistics must guide model building. For example, it is important to give statistically significant random samples are training data so ML algorithms won't find patterns that wouldn't be there in other datasets. In addition, if the training set is too small, we may reach inaccurate conclusions due to the Law of Large Numbers.

### Gradient Descent

Optimizing functions like:
<br>
![h of x equals theta 0 plus theta 1 times x](https://uploads.toptal.io/blog/image/444/toptal-blog-image-1407508963294.png)
<br>
are often the subject of simple univariate linear regression problems. However, as we approach more complicated predictor functions such as:
<br>
![Four dimensional equation example](https://uploads.toptal.io/blog/image/456/toptal-blog-image-1407511674278.png)
<br>
The question of optimization now becomes more complex. The above function, for example, takes input in four dimensions, as opposed to one. In order to solve more complex problems, approaches such as gradient descent must be used to "minimize wrongness".

Gradient descent uses an iterative process to make sure that values like ![theta 0](https://uploads.toptal.io/blog/image/445/toptal-blog-image-1407508999266.png) and ![theta 1](https://uploads.toptal.io/blog/image/446/toptal-blog-image-1407509052417.png) are getting closer to the real values with each step. We can use calculus to "measure wrongness" are decide the appropriate next step.

The wrongness measure is known as the **cost function** (a.k.a., **loss function**), <br>
![J of theta](https://uploads.toptal.io/blog/image/458/toptal-blog-image-1407512229600.png)<br>. Here, ![theta](https://uploads.toptal.io/blog/image/461/toptal-blog-image-1407512699363.png) just represents all the coefficients being used in our predictor. In the univariate case, ![theta](https://uploads.toptal.io/blog/image/461/toptal-blog-image-1407512699363.png) is really the pair of ![theta 0](https://uploads.toptal.io/blog/image/445/toptal-blog-image-1407508999266.png) and ![theta 1](https://uploads.toptal.io/blog/image/446/toptal-blog-image-1407509052417.png). ![J of theta 0 and theta 1](https://uploads.toptal.io/blog/image/460/toptal-blog-image-1407512532218.png) gives us a mathematical measurement of how wrong our predictor is when it uses the given values of ![theta 0](https://uploads.toptal.io/blog/image/445/toptal-blog-image-1407508999266.png) and ![theta 1](https://uploads.toptal.io/blog/image/446/toptal-blog-image-1407509052417.png) – this is, of course, in reference to our already known value.

The choice of a cost function is important, because being "wrong" can mean different things in different contexts. For a linear situation, the linear least squares function is a well-established cost function:

![Cost function expressed as a linear least squares function](https://uploads.toptal.io/blog/image/473/toptal-blog-image-1407783702580.png)

In least squares, the "penalty" for a bad guess goes up quadratically with the difference between the guess and the known, correct answer. The cost function computes an average penalty over all the available training examples.

We can summarize our goal in this: to use calculus to find ![theta 0](https://uploads.toptal.io/blog/image/445/toptal-blog-image-1407508999266.png) and ![theta 1](https://uploads.toptal.io/blog/image/446/toptal-blog-image-1407509052417.png) for our predictor `h(x)` such that our cost function![J of theta 0 and theta 1](https://uploads.toptal.io/blog/image/460/toptal-blog-image-1407512532218.png) is as small as possible.

Consider the following plot:
<br>
![gradient descent](https://uploads.toptal.io/blog/image/125327/toptal-blog-image-1517837664732-02321506a70221b9012cbf5770bdc53f.png)
<br>
The axes include ![theta 0](https://uploads.toptal.io/blog/image/445/toptal-blog-image-1407508999266.png) , ![theta 1](https://uploads.toptal.io/blog/image/446/toptal-blog-image-1407509052417.png), and  ![J of theta 0 and theta 1](https://uploads.toptal.io/blog/image/460/toptal-blog-image-1407512532218.png).

We see that the graph has a bowl-like curve. We notice that the bottom of the bowl represents the lowest cost (least wrongness) our predictor can give us based on training data. In physical terms, we want to "roll down the hill" and find ![theta 0](https://uploads.toptal.io/blog/image/445/toptal-blog-image-1407508999266.png) and ![theta 1](https://uploads.toptal.io/blog/image/446/toptal-blog-image-1407509052417.png) corresponding to this point.

McCrea provides a concise explanation of the calculus behind this:

> What we do is take the gradient of ![J of theta 0 and theta 1](https://uploads.toptal.io/blog/image/460/toptal-blog-image-1407512532218.png), which is the pair of derivatives of ![J of theta 0 and theta 1](https://uploads.toptal.io/blog/image/460/toptal-blog-image-1407512532218.png) (one over ![theta 0](https://uploads.toptal.io/blog/image/445/toptal-blog-image-1407508999266.png) and one over ![theta 1](https://uploads.toptal.io/blog/image/446/toptal-blog-image-1407509052417.png)). The gradient will be different for every different value of ![theta 0](https://uploads.toptal.io/blog/image/445/toptal-blog-image-1407508999266.png) and ![theta 1](https://uploads.toptal.io/blog/image/446/toptal-blog-image-1407509052417.png), and tells us what the “slope of the hill is” and, in particular, “which way is down”, for these particular ![theta](https://uploads.toptal.io/blog/image/461/toptal-blog-image-1407512699363.png)s. For example, when we plug our current values of ![theta](https://uploads.toptal.io/blog/image/461/toptal-blog-image-1407512699363.png) into the gradient, it may tell us that adding a little to ![theta 0](https://uploads.toptal.io/blog/image/445/toptal-blog-image-1407508999266.png) and subtracting a little from ![theta 1](https://uploads.toptal.io/blog/image/446/toptal-blog-image-1407509052417.png) will take us in the direction of the cost function-valley floor. Therefore, we add a little to ![theta 0](https://uploads.toptal.io/blog/image/445/toptal-blog-image-1407508999266.png), and subtract a little from ![theta 1](https://uploads.toptal.io/blog/image/446/toptal-blog-image-1407509052417.png), and voilà! We have completed one round of our learning algorithm. Our updated predictor, `h(x)` = ![theta 0](https://uploads.toptal.io/blog/image/445/toptal-blog-image-1407508999266.png) + ![theta 1](https://uploads.toptal.io/blog/image/446/toptal-blog-image-1407509052417.png)x, will return better predictions than before. Our machine is now a little bit smarter.

This overall process of updating the ![theta](https://uploads.toptal.io/blog/image/461/toptal-blog-image-1407512699363.png)s and calculating the current gradient is known as gradient descent. 

Josh Starmer on behalf of StatQuest has a great video on the gradient descent that can be found [here](https://youtu.be/sDv4f4s2SB8). Starmer goes into more mathematical detail on gradient descent.

We can summarize the gradient descent algorithm to be:

 1. Take the derivative of the loss function for each parameter in it. This is taking the gradient of the loss function.
 2. Pick random values for the parameters.
 3. Plug the parameter values into the derivatives (the gradient)
 4. Calculate the step sizes where step size = slope x learning rate
 5. Calculate the new parameters where the new parameter = old parameter - step size
 6. Repeat steps 3-5 until the step size is very close to 0 or the algorithm reaches the maximum amount of steps.

### Classification vs Regression

In supervised ML, there are two main machine learning categories: classification and regression.

 - **Regression**: The algorithm predicts a value that falls in the continuous spectrum. Questions such as "How much" or "How many" are relevant here
 - **Classification**: The algorithm predicts a categorical prediction such as "is this tumor cancerous"

#### Classification Machine Learning Theory

The only differences between the underlying machine learning theory above and theory specific to classification are the design of the predictor `h(x)` and the design of the cost function ![J of theta](https://uploads.toptal.io/blog/image/458/toptal-blog-image-1407512229600.png).

We want our predictor to make a guess between 0 and 1, where values closer to 1 represent a high degree of confidence about a particular classification. The sigmoid function can be helpful in this:
<br>
![sigmoid function](https://uploads.toptal.io/blog/image/125331/toptal-blog-image-1517837793467-920401fca5df9ae5748522a1d82c746f.png)
<br>
The sigmoid function is often called `g(z)`, where `z` is some representation of our inputs and coefficients, such as:
<br>
![enter image description here](https://uploads.toptal.io/blog/image/469/toptal-blog-image-1407513632307.png)
<br>
Our predictor now becomes:
<br>
![enter image description here](https://uploads.toptal.io/blog/image/474/toptal-blog-image-1407783785110.png)
<br>
The sigmoid function essentially transforms output from `z` to produce a value between 0 and 1 that is useful for classification.

McCrea goes into some detail about the logic behind the design of the cost function:

> Again we ask “what does it mean for a guess to be wrong?” and this time a very good rule of thumb is that if the correct guess was 0 and we guessed 1, then we were completely and utterly wrong, and vice-versa. 
> 
> Since you can’t be more wrong than absolutely wrong, the penalty in this case is enormous. Alternatively if the correct guess was 0 and we guessed 0, our cost function should not add any cost for each time this happens.
> 
> If the guess was right, but we weren’t completely confident (e.g.  `y = 1`, but  `h(x) = 0.8`), this should come with a small cost, and if our guess was wrong but we weren’t completely confident (e.g.  `y = 1`  but  `h(x) = 0.3`), this should come with some significant cost, but not as much as if we were completely wrong.

The log function captures this behavior:
<br>
![enter image description here](https://uploads.toptal.io/blog/image/471/toptal-blog-image-1407513738977.png)
<br>
And the cost function ![J of theta](https://uploads.toptal.io/blog/image/458/toptal-blog-image-1407512229600.png) gives us the average cost over all of our training examples.

Thus, we can see that the predictor `h(x)` and the cost function ![J of theta](https://uploads.toptal.io/blog/image/458/toptal-blog-image-1407512229600.png) differ between regression and classification, but gradient descent in both cases still works fine.

## A note about Neural Networks

Neural networks are a powerful way to produce machine learning models with an enormous amount of input data.

A good primer on how neural networks can work is 3Blue1Brown's video on neural networks and deep learning [here](https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown).

## Unsupervised Learning

The goal of unsupervised learning is to uncover patterns and relationships within data. In this case, there are no training samples. Some popular approaches include clustering (ex. k-means), dimensionality reduction (ex. principle component analysis) and more.