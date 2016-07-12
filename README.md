For training networks for ASR.

Contains class `Network.

# Example program
	
    import tensorflow as tf
    from NN import Network, pretrain_network
    import h5py

    def main():
         data_train_file = '...'
         data_val_file = '...'

	 shape = [440, 1024, 1024, 1024, 1024, 1024, 1024, 2000]
	 
	 
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


# pretrain_network

The DBN paper intimidated me so I thought maybe I could try pretrain a network using other methods, such as used in [1](http://research.microsoft.com/pubs/157341/FeatureEngineeringInCD-DNN-ASRU2011-pub.pdf) or [2](https://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf)

Results so far have been disappointing. Using a kaldi network of the same size and DBN pretraining achieves an 11-12% WER, I'm still trying to get past 16. I believe the underlying problem is I've got 10 hours of training data (aurora4), the linked papers have an order of magnitude more; per [3](http://research.google.com/pubs/pub38131.html) the more data you have the less pretraining matters.

# hdf5 data format

This only works with the data formatted correctly. The advantage is very low RAM usage.
It's not hard to do, but to make it as simple as possible here's an example. It is assumed that the features and targets are numpy arrays in `f` and `t`.

	import h5py

	# This is where we want out data to end up in.
	data_fname = '...' 

	new_data_file = h5py.File(data_fname, 'w')
	
	# Note data must use these names.
	new_data_file.create_dataset('feats', data=f)
	new_data_file.create_dataset('targs', data=t)

	new_data_file.close()

	# That's it !
	

	
