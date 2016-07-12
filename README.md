For training networks for ASR. Contains a `Network` class that's an OO implementation of tensorflow. Example program below.

# Network

This class uses Tensorflow to create a Neural Network. Trained networks can be saved and restored.
Assumes training data is in hdf5 file for efficient memory usage.
(For kaldi) Can, given a file of utterance IDs with their feature vectors, output result into a file that kaldi can
	process (specifically `latgen-faster-mapped` from nnet1). 

Inside python use `help(Network)` to bring up the help text showing available functions. Summary of implementation:

Fixed:

	Elu activation function, Adam SGD, weight init using `tf.truncated_normal(stddev=1/(input*1.41))` (bias=0), softmax.

Variable:

	Constant learning rate or adaptive, dropout, L2 regularization.

A lot of smaller features (as in useful functions etc.) are also there.

Possibility to split network into separate streams (that then join) is being worked on (was pretty much done, fallen by the wayside for the moment).

# Example

	from NN import Network
	
	data_train = '....hdf5'
	data_val = '....hdf5'
	pretrain_param_save_file = '...'

	NN = Network([520, 1024, 1024, 1024, 1024, 1024, 1024, 2000], pretrain=True,
		pretrain_params_dict={'data_train': data_train, 'data_val': data_val,		
					'epochs': 50, 'batch_size': 512, 'eta': 1e-4,
					'kp_prob': 0.8, 'lam': 0, 
					'save_file': pretrain_param_save_file})

	NN.train(data_train, 20, 512, 1e-4, val_file=data_val, eta_policy='adaptive', 
							kp_prob=0.8, score_pt_d=20)

	model_save_file = '...'
	NN.save(model_save_file)

	NN.stop()

# pretrain_network

The DBN paper intimidated me so I thought maybe I could try pretrain a network using other methods, such as used in [1](http://research.microsoft.com/pubs/157341/FeatureEngineeringInCD-DNN-ASRU2011-pub.pdf) or [2](https://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf)

Results so far have been disappointing. Using a kaldi network of the same size and DBN pretraining achieves an 11-12% WER, I'm still trying to get past 16. I believe the underlying problem is I've got 10 hours of training data (aurora4), the linked papers have an order of magnitude more; per [3](http://research.google.com/pubs/pub38131.html) the more data you have the less pretraining matters.

# hdf5 data format

Training only works with the data formatted correctly. The advantage is very low RAM usage.
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
	
# Dependencies

    Python 2.7

    Tensorflow, h5py, numpy, subprocess, re, datetime
	
