---
layout: page
title: "Optimization Algorithms in Bird Classification"
permalink: /project/
---
<h2>By Anna McAuliffe</h2>

<h1>Project Description</h1>

Optimization algorithms are a crucial part of any neural training process, and the performance of an optimization algorithm can strongly affect the convergence, speed, and accuracy of the model in question. In the realm of convolutional neural networks (or CNNs), there are a couple of algorithms that come out as the most popular or effective: Stochastic Gradient Descent and Adam. 

SGD is an improvement on standard gradient descent, using randomized samples from the overall dataset to reduce performance cost and reduce overfitting to some degree. 

Adam, on the other hand, is a newer but very popular alternative that combines the techniques of AdaGrad (Adaptive Gradient Algorithm) and RMSProp (Root Mean Square Propagation). 

In this project, I will test how effective these two optimization algorithms, as well as a few others that drew my interest, perform against the Kaggle Bird Species Classification dataset.

<h1>Neural Model Construction</h1>

I wanted to ensure that as much of the neural model's architecture remained the same throughout this whole experiment. With this in mind, I used the pretrained resnet18 model available for PyTorch. I also tried to keep as much of the example training method from the in-class tutorials as I possibly could, to ensure consistency in my experiments, but there were necessary changes to make for some optimization algorithms (which I will detail in their result sections). 

While it is possible that these differences will disrupt my final results, I don't think there is a feasible way to avoid them, as each optimizer uses different parameters for different reasons. Learning rate, for instance, has a vastly different effect for most of the optimizers, so it had to be manually adjusted at times to allow the optimizers to perform to their strengths. Again, all of these adjustments will be noted below.

<h1>Hyperparameters</h1>

For the most part, the following hyperparameters remained consistent throughout all experiments:

*num_epochs = 7* (True for all trials)

*lr = 0.01* (True for all applicable trials except Adam)

*momentum = 0.9* (True for all applicable trials)

*decay = 0.0005* (True for all applicable trials)

<h1>Stochastic Gradient Descent</h1>

Perhaps the most obvious solution to the optimizer debate, SGD has been a mainstay of CNN optimization for a long time, and for good reason. It maintains many of the strengths of Gradient Descent without the extreme computational complexity that GD optimization creates, while also mitigating risks of overfitting through its random sampling approach. This was the default optimizer given in both tutorials 3 and 4, and is widely supported by most articles on ML that I've been able to find.

Although converging a bit slower than the other optimization algorithms that I tested, SGD converged to a much lower loss value than the other algorithms:

![SGD Loss over Iterations](/assets/SGD_loss.png)

SGD also proved to have the greatest testing accuracy of the optimizers, reaching as far as **0.66050 testing accuracy**.

SGD does have some issues, namely its fixed learning rate for all parameters and its high noise in the training process, but in my experiments, it proved to be the most consistent optimization function regardless. I attribute this to its ease of use, simplicity, and preventative measures against overfitting. Overfitting became a serious issue for other algorihms I explored, but SGD had little issue with it.

<h1>Adam</h1>

Most of my understanding on Adam comes from the PyTorch Adam documentation as well as Kingma and Ba's 2015 study "Adam: A Method for Stochastic Optimization", cited below.

Adam is best known for its ability to compute its own learning rates for different parameters and the state of training (literally short for adaptive moment estimation), which was popularized by optimizers like AdaGrad and RMSProp (more on those below). Adam is effective in combining the advantages of these two optimizers, allowing for efficient performance against sparse gradients and built-in annealing that adapts to the learning process.

While these features theoretically should reduce overfitting (this is one of the biggest benefits of annealing!), it didn't appear to do so in my experimentation. Adam's optimization did indeed converge much more quickly than SGD, but it would then spend most of the training epoch gradually climbing up in loss, before "learning" between epochs:

![Adam Loss over Iteration](/assets/adam_loss_001.png)

I would assume that this indicates a greater efficacy over more epochs, but at the same time, the last few epochs indicate nearly climbing all the way back up to the previous epoch's minimum loss, which means overfitting is likely occuring. This is also indicated in the testing results, where Adam performed reasonably worse than SGD, coming in with **0.48800 accuracy**.

I would likely attribute some of Adam's failings to the hyperparameters not suiting its performance, as it seems like it would benefit from more epochs to a greater degree than the other optimization algorithms. That said, SGD's success the same period indicates that perhaps Adam's efficient training benefits are perhaps not as important with this dataset or model, and that SGD is better suited here than this newer, more-optimized optimizer.

It is worth noting that I had some serious issues with Adam at first, as I tried to ensure that the initial learning rate was the same as in the SGD trial. However, as the algorithm [found in Adam's documentation shows](https://arxiv.org/abs/1412.6980), Adam handles learning rates quite differently than SGD, so I decided to let it use its default learning rate of 0.001 instead of the set hyperparameter of 0.01. This significantly improved its performance, and indicated to me that perhaps learning rate isn't as good of a control as I initially anticipated, as every optimizer uses it a little differently.

<h1>AdaDelta</h1>

AdaDelta isn't a pivotal or unusually popular optimizer, unlike the first two that I tested, but its unique properties stuck out to me as something I wanted to experiment with. AdaDelta is similar to other Ada-like algorithms in that it adapts its learning rate based on the state of the model, but AdaDelta is unique in that it only bases its adaptations on ["a moving window of gradient updates"](https://keras.io/api/optimizers/adadelta/#:~:text=Adadelta%20is%20a%20more%20robust,many%20updates%20have%20been%20done.) as opposed to the standard strategy of reviewing all previous gradients. This is particularly interesting as it prevents the problem that many of these Ada- algorithms have of training becoming less effective as more updates occur.

I was particularly drawn to AdaDelta because it uniquely doesn't ask for an initial learning rate at all, with a default learning rate of 1 that it adjusts as time goes on. While it doesn't necessarily have the streamlined optimizations of Adam or the thoroughness of SGD, I was interested to see how such a unique algorithm performed in this model:

![AdaDelta Loss over Iteration](/assets/adadelta.png)

<h1>Works Cited</h1>

"PyTorch Adam Documentation". [https://pytorch.org/docs/stable/generated/torch.optim.Adam.html](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)

"Keras Adadelta Documentation". [https://keras.io/api/optimizers/adadelta/#:~:text=Adadelta%20is%20a%20more%20robust,many%20updates%20have%20been%20done.](https://keras.io/api/optimizers/adadelta/#:~:text=Adadelta%20is%20a%20more%20robust,many%20updates%20have%20been%20done.)

Kinga, Diederik P. and Ba, Jimmy. (2015). "Adam: A Method for Stochastic Optimization". *ICLR 2015*. [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

