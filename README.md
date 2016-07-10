For training networks for ASR.

Contains class `Network` and function `pretrain_network`.

# Example program
	
    import tensorflow as tf
    from NN import Network, pretrain_network
    import h5py

    def main():
         data_train_file = '...'
         data_val_file = '...'

	 shape = [440, 1024, 1024, 1024, 1024, 1024, 2000]
	 
	 
	 with tf.Graph().as_default():
	 	pretrain_network(shape, data_train_file, 10, 512, val_file=data_val_file, name='params_file')
	 	
	 with tf.Graph().as_default():	
	 	
	 	NN = Network(shape, pretrain=True, restore_p = 'params_file')
	 	
	 	NN.train(data_train_file, 20, 512, 1e-4, val_file = data_val_file, lmbda = 1e2, eta_policy='adaptive')
	 	
	 	feats_test, targs_test = get_test_data()
	 	
	 	print(NN.score(feats_test, targs_test))
	 	
	 	NN.save('trained_model')
	 	
	 	NN.stop()


# Network

This class uses Tensorflow to create a Neural Network. Trained networks can be saved and restored.
(For kaldi) Can, given a file of utterance IDs with their feature vectors, output result into a file that kaldi can
	process (specifically `latgen-faster-mapped` from nnet1). Assumes training data is in hdf5 file for efficient memory usage.

Inside python use `help(Network)` to bring up the help text showing available functions. Summary of implementation:

Fixed:

	Elu activation function, Adam SGD, weight init using `tf.truncated_normal(stddev=0.1)`, softmax.

Variable:

	Constant learning rate or adaptive, dropout, L2 regularization.

Possibility to split network into separate streams (that then join) is being worked on.

If `pretrain_network` is used before separate using `tf.Graph().as_default()`.

# pretrain_network

Uses discriminative training similar to the one used in this paper: http://research.microsoft.com/pubs/157341/FeatureEngineeringInCD-DNN-ASRU2011-pub.pdf

Methods used here are also being tested: https://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf

Will be updated to use the better one depending on results.

Again use `help(pretrain_network)` for explanation on use.


