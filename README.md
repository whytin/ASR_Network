For training networks for ASR.

Contains class `Network` and function `pretrain_network`.

# Network

This class uses Tensorflow to create a Neural Network. Trained networks can be saved and restored.
(For kaldi) Can, given a file of utterance IDs with their feature vectors, output result into a file that kaldi can
	process (specifically `latgen-faster-mapped` from nnet2).

Inside python use `help(Network)` to bring up the help text showing available functions. Summary of implementation:

Fixed:
	Elu activation function, Adam SGD, weight init using `tf.truncated_normal(stddev=0.1)`, softmax.
Variable:
	Constant learning rate or adaptive, dropout, L2 regularization.

Possibility to split network into separate streams (that then join) is being worked on.

If `pretrain\_network` is used before separate using `tf.Graph().as\_default()`.

# pretrain\_network

Uses discriminative training similar to the one used in this paper: http://research.microsoft.com/pubs/157341/FeatureEngineeringInCD-DNN-ASRU2011-pub.pdf

Again use `help(pretrain_network)` for explanation on use.

