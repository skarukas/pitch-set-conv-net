# pitch-set-conv-net
A small custom 1D convolutional neural network for learning pitch set transformations. All parts of the network are implemented from scratch in MATLAB, and this was mainly for practice.

Without explicit programming, the network learns a task, such as transposing a pitch set up 2 steps or inverting it around a certain pitch, with 100% accuracy. The 1D circular convolutions introduce transpositional/translational invariance, meaning a network trained to infer properties from a C major chord will generalize very well to a D major chord. 

These are toy examples, but they show interesting properties, of both pitch sets and the ability of convolutional filters to transform them. I'll be working on testing this network on more complex operations such as chord recognition and adaptive just intonation in the future.

#### Some things implemented in these files:

- Various activation functions
- Circular/Periodic 1D Convolution
- Gradient Descent
- Accelerated Descent
- Momentum Gradient Descent
- AdaGrad
- Convoluional Neural Network 
  - one layer of 1D circular convolution (6 filters of size 12), no pooling
  - one fully connected layer 1x6, applied to each pitch class to maintain transpositional invariance

Also implemented is a visualization of the algorithm's convergence with respect to its parameters, using PCA dimensionality reduction to reduce the 162-dimensional parameter space to two latent variables.
