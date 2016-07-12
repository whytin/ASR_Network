from __future__ import division
import tensorflow as tf
import numpy as np
import subprocess
from datetime import datetime
import re
import h5py


def ini_weight_var(shape, name=None):
	initial=tf.truncated_normal(shape, stddev=0.0001)
	return tf.Variable(initial, name=name, trainable=True)


def ini_bias_var(shape, name=None):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial, name=name, trainable=True)


def batch_gen(f, t, div_bs, num_iters, len_feat, div):
	batch_idx = np.random.permutation(num_iters)*div_bs
	for i in xrange(0, len(batch_idx)-div, div):
		bf, bt = np.zeros((div, div_bs, len_feat)), np.zeros((div, div_bs,))
		for j in xrange(div):
			start = batch_idx[i+j]
			end = start+div_bs
			bf[j], bt[j] = f[start:end], t[start:end]
		bf = np.concatenate(bf, axis=0)
		bt = np.concatenate(bt, axis=0)
		yield bf, bt


class Network:
	'''

	This class uses Tensorflow to create a Neural Network. Trained networks can be saved and restored. Assumes training data is in hdf5 file
	for efficient memory usage.
	(For kaldi) Can, given a file of utterance IDs with their feature vectors, output result into a file that kaldi can
		process.

	More options are planned.

	Hardcoded:
		Elu activation functions.
		SGD-method is Adam.
		Weight initialization uses tf.truncated_normal(stddev=0.1).
		Softmax.
	Variable:
		Dropout.
		Regularization (L2).
		Constant learning rate or adaptive (/2 if cost function not decreased after (default) 5 epochs).


	Ex:
		NN = Network([50,100,10])	- 	Neural Network with input layer of 50, one hidden layer of 100, and softmax
			layer of 10 neurons.


	! Don't forget to stop the session with <Network_var>.stop() to free up resources when done. Will keep running otherwise !

	'''

	def init_layers(self, split):
		self.weights = []
		self.biases = []
		self.a = []
		self.a_drop = []

		if not split:
			for i, len_layer in enumerate(self.shape[1:-1]):
				self.weights.append(ini_weight_var([self.shape[i], len_layer], 'w'+str(i)))
				self.biases.append(ini_bias_var([len_layer], 'b'+str(i)))
				if i==0:
					self.a.append(tf.nn.elu(tf.matmul(self.input_layer, self.weights[i]) + self.biases[i]))
					self.a_drop.append(tf.nn.dropout(self.a[0], self.kp_prob))
				else:
					self.a.append(tf.nn.elu(tf.matmul(self.a_drop[i-1], self.weights[i]) + self.biases[i]))
					self.a_drop.append(tf.nn.dropout(self.a[i], self.kp_prob))

			# Output.
			self.weights.append(ini_weight_var([self.shape[-2], self.shape[-1]], 'wl'))
			self.biases.append(ini_bias_var([self.shape[-1]], 'bl'))

			self.output = tf.matmul(self.a_drop[-1], self.weights[-1]) + self.biases[-1]
			self.softm_output = tf.nn.softmax(self.output)

		else:
			switch = True
			self.inp_layers = tf.split(1, split[0], self.input_layer)
			for i, len_layer in enumerate(self.shape[1:-1]):
				div = split[i]
				if div != 1:
					len_piece1 = int(self.shape[i]/div)
					len_piece2 = int(len_layer/div)
					self.weights.append([])
					self.biases.append([])
					self.a.append([])
					self.a_drop.append([])
					for j in xrange(div):
						self.weights[i].append(ini_weight_var([len_piece1, len_piece2]))
						self.biases[i].append(ini_weight_var([len_piece2]))
						if i==0:
							self.a[i].append(tf.nn.elu(tf.matmul(self.inp_layers[j], self.weights[i][j]) + self.biases[i][j]))
							self.a_drop[i].append(tf.nn.dropout(self.a[i][j], self.kp_prob))
						else:
							self.a[i].append(tf.nn.elu(tf.matmul(self.a_drop[i-1][j], self.weights[i][j]) + self.biases[i][j]))
							self.a_drop[i].append(tf.nn.dropout(self.a[i][j], self.kp_prob))
				else:
					self.weights.append(ini_weight_var([self.shape[i], len_layer]))
					self.biases.append(ini_bias_var([len_layer]))
					if switch:
						switch = False
						self.joined = tf.concat(1, self.a_drop[i-1])
						self.a.append(tf.nn.elu(tf.matmul(self.joined, self.weights[i]) + self.biases[i]))
						self.a_drop.append(tf.nn.dropout(self.a[i], self.kp_prob))
					else:
						self.a.append(tf.nn.elu(tf.matmul(self.a_drop[i-1], self.weights[i]) + self.biases[i]))
						self.a_drop.append(tf.nn.dropout(self.a[i], self.kp_prob))

			# Output
			self.weights.append(ini_weight_var([self.shape[-2], self.shape[-1]]))
			self.biases.append(ini_bias_var([self.shape[-1]]))
			self.output = tf.matmul(self.a_drop[-1], self.weights[-1]) + self.biases[-1]
			self.softm_output = tf.nn.softmax(self.output)

	def __init__(self, shape, pretrain=False, split=False, pretrain_params_dict=None, use_fraction_gpu=1):
		'''
		Creates graph and starts session. Can load weights from a saved file created by the function `pretrain_network`.
		Args:
			shape: 					Shape of network. Expects list.

			pretrain: 				Load saved weights or not.
				default - False

			pretrain_params_dict: 	Dictionary containing: {'data_train', 'data_val', 'epochs',
										'batch_size', 'eta', 'kp_prob', 'save_file'}
									'save_file' - File to save pretrained paramaters that will
										be loaded in.
									If other keys unclear read the docs of the `pretrain_network`
									or `train` function.

				default - None

			! Still being worked on:
			split:					Divisor with which to split network by. Ex. for network [100,1000,1000,1000,500] -> (10,10,10,1,1)
				default - False

			use_fraction_gpu: 		Use a fraction of instead of all the available GPU memory.
				default - 1

		'''

		if pretrain:
			with tf.Graph().as_default():
				pretrain_network(shape, pretrain_params_dict['data_train'], pretrain_params_dict['epochs'],
					 pretrain_params_dict['batch_size'], pretrain_params_dict['eta'],
					 pretrain_params_dict['data_val'], pretrain_params_dict['kp_prob'], pretrain_params_dict['save_file'])

		self.shape = np.asarray(shape)

		# Input on 'train time'.
		self.eta = tf.placeholder("float")
		self.input_layer = tf.placeholder("float",shape=[None,self.shape[0]])
		self.kp_prob = tf.placeholder("float")
		self.lmbda = tf.placeholder("float")
		self.targets = tf.placeholder("int64", shape=[None,])

		self.init_layers(split)

		# Regularization.
		if not split:
			self.param_sum = tf.add_n([tf.nn.l2_loss(w_mat) for w_mat in self.weights]) #+ tf.add_n([tf.nn.l2_loss(b_v) for b_v in self.biases])
			self.param_num = np.sum([f*s for f, s in zip(self.shape[:-1], self.shape[1:])]) #+ np.sum(self.shape[1:])
		else:
			jnd_idx = split.index(1)

			self.param_sum = tf.add_n([tf.nn.l2_loss(self.weights[idx])+tf.nn.l2_loss(self.biases[idx])
																for idx in xrange(jnd_idx, len(split)-1)])
			self.param_num = np.sum([f*s for f, s in zip(self.shape[jnd_idx:-1], self.shape[jnd_idx:])]) + np.sum(self.shape[jnd_idx:])

		# Cross entropy function (1: Individual cross entropy error, 2: Total cross entropy with reg.).
		self.error = tf.nn.sparse_softmax_cross_entropy_with_logits(self.output, self.targets)
		self.cost_f = tf.reduce_sum(self.error) + self.lmbda*tf.div(self.param_sum, self.param_num)
		self.cost_m = tf.reduce_mean(self.error) + self.lmbda*tf.div(self.param_sum, self.param_num)

		# Adam SGD.
		self.train_adam = tf.train.AdamOptimizer(self.eta).minimize(self.cost_m)

		self.saver = tf.train.Saver()

		self.get_acc = tf.reduce_mean(tf.cast(tf.equal( tf.argmax(self.softm_output, 1), self.targets ) ,"float"))

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=use_fraction_gpu)

		if pretrain:

			params_dict = {}
			for i in xrange(len(self.weights)):
				if i == len(self.weights)-1:
					params_dict['wl'] = self.weights[-1]
					params_dict['bl'] = self.biases[-1]
				else:
					params_dict['w'+str(i)] = self.weights[i]
					params_dict['b'+str(i)] = self.biases[i]

			get_pt = tf.train.Saver(params_dict)
			self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

			get_pt.restore(self.sess, pretrain_params_dict['save_file'])

			in_v = [v for v in tf.all_variables() if v not in tf.trainable_variables()]
			self.sess.run(tf.initialize_variables(in_v))

		else:
			self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
			self.sess.run(tf.initialize_all_variables())

		print("Session running.")

	def train(self, data_file, epochs, batch_size, eta, kp_prob=1, eta_policy='const', lmbda=0, val_file='None',
				eta_chk_pt=3, score_pt_d=0, partial_scoring = True):
		'''
		Trains network. Minimizes the mean of cross-entropy for softmax outputs otherwise squared error.
		Args:
			data_file: 	 	hdf5 file containing feature and target datasets. Features must be in dataset 'feats', targets in dataset 'targs'.
			epochs:			Number of iterations that SGD should perform.
			batch_size:		Number of (feat,targ) pairs to use per SGD iteration.
			eta:			Learning rate.

			kp_prob:		Proportion of neurons that should not be dropped (kept).
				default	- 1

			adapt_eta:		'const' -> constant learning rate
							'adaptive' -> adaptive learning rate (/2 if cost function not decreased after 3e3 epochs)
				default - 'const'

			lmbda:			Regularization multiplier.
				default - 0

			val_file: 	 	hdf5 file containing feature and target datasets. Features must be in dataset 'feats', targets in dataset 'targs'.
				default - 'None'

			eta_chk_pt:		Number of epochs at which to check cost to decrease or keep constant the learning rate.
				default - 3

			score_pt_d: 	Number of times to display accuracy/error.
				default - 10

			partial_scoring: Takes 50 000 samples from the training set and uses them for scoring/error evaluation. If
							set to `False`, the entire set will be used.
				default - True

		Data is shuffled each epoch and then stepped through with via batch_size. Score and/or error is
		output 10 times by default in a training session.
		'''

		train_opt = self.train_adam

		df = h5py.File(data_file,'r')
		feats, targs = df['feats'], df['targs']

		if partial_scoring:
			sc_f, sc_t = feats[:50000], targs[:50000]
			len_feat = len(sc_f[0])
		else:
			sc_f, sc_t = feats, targs
			len_feat = len(sc_f[0])

		if score_pt_d == 0:
			score_pt = int(epochs/10)
		else:
			score_pt = int(epochs/score_pt_d)

		if val_file == 'None':
			val_data = False
		else:
			val_data = True
			val_d = h5py.File(val_file, 'r')
			val_feats, val_targs = val_d['feats'][:50000], val_d['targs'][:50000]

		print("Beginning training.")

		if eta_policy=='adaptive':
			last_cost = 1e20

		if eta_policy == 'ES':
			epochs = 100
			original_eta = eta

		div = 2
		div_batch = int(batch_size/div)
		num_iters = int(len(feats)/div_batch)

		tt = datetime.now()
		for epoch in xrange(epochs):
			for batch_f, batch_t in batch_gen(feats, targs, div_batch, num_iters, len_feat, div):

				self.sess.run(train_opt, feed_dict={self.input_layer: batch_f, self.targets: batch_t,
														  self.eta: eta, self.kp_prob: kp_prob, self.lmbda: lmbda})

			if eta_policy == 'const':
				if epoch%score_pt == 0:
					if val_data:
						print("Training error {2}, Training score {0}, Val score {1}".format(self.score(sc_f, sc_t), self.score(val_feats, val_targs), self.cost_mean(sc_f, sc_t)))
					else:
						print("Training score {0}, training error {1}".format(self.score(sc_f, sc_t), self.cost_mean(sc_f, sc_t)))

			elif eta_policy == 'adaptive':
				if epoch%eta_chk_pt == 0:
					new_cost = self.cost_mean(feats, targs)
					print("Training error {0}".format(new_cost))
					if new_cost > last_cost:
						eta /= 2
					last_cost = new_cost
				if epoch%score_pt:
					if val_data:
						print("Training score {0}, Val score {1}".format(self.score(sc_f, sc_t), self.score(val_feats, val_targs)))
					else:
						print("Training score {0}".format(self.score(sc_f, sc_t)))

			elif eta_policy == 'ES':
				if epoch % eta_chk_pt == 0:
					new_cost = self.cost_mean(feats, targs)
					print("Training error {0}".format(new_cost))
					if new_cost > last_cost:
						eta /= 2
					last_cost = new_cost

				if val_data:
					print("Training score {0}, Val score {1}".format(self.score(sc_f, sc_t),
																	 self.score(val_feats, val_targs)))
				else:
					print("Training score {0}".format(self.score(sc_f, sc_t)))
				if eta == original_eta/8:
					break

			else:
				print("No output: `eta_policy` must equal to 'const' or 'adaptive'.")

		df.close()
		if val_data:
			val_d.close()

		print("Training duration: {0}".format(datetime.now()-tt))

	def cost_sum(self, feats, targs):
		'''
		Input: Feature vectors and their targets. Returns sum of the cost function outputs.
		'''
		return self.sess.run(self.cost_f, feed_dict = {self.input_layer:feats,
								self.targets:targs,self.kp_prob:1,self.lmbda:0})

	def cost_mean(self, feats, targs):
		'''
		Input: Feature vectors and their targets. Returns mean of the cost function outputs
		'''
		return np.mean(self.sess.run(self.error, feed_dict = {self.input_layer:feats,
								self.targets:targs,self.kp_prob:1,self.lmbda:0}))

	def score(self, feats, targs):
		'''
		Input: Feature vectors and their targets. Returns proportion of outputs that fit targets.
		'''
		return self.sess.run(self.get_acc, feed_dict = {self.input_layer:feats,
							self.targets:targs,self.kp_prob:1,self.lmbda:0})

	def forward_pass(self, inp, apply_log=False):
		'''
		Input: Feature vectors. Returns outputs.
		'''
		if apply_log is True:
			return np.log(self.sess.run(self.softm_output,feed_dict = {self.input_layer:inp, self.kp_prob:1})+1e-15)
		else:
			return self.sess.run(self.softm_output,feed_dict = {self.input_layer:inp, self.kp_prob:1})

	def feats_extractor(self, feats_file):
		pattern = re.compile('[a-zA-Z]')
		key, mat = '', []
		started = False
		with open(feats_file) as f:
			for line in f:
				if bool(pattern.search(line[:10])):
					if started:
						yield key, mat
						mat = []
					key = line[:-2].strip()
					started = True
				else:
					mat.append([float(s) for s in line[:-2].strip().split()])


	def output_for_kaldi(self, feats_file, SNR=None, splicing=0):
		'''
		Takes a txt file of the form

			<utterance-ID> [
			<feature vector>
			...
			<feature vector> ]
			<utterance-ID> [
			...

		extracts the utterances and corresponding feature vectors that match a SNR, calculates the outputs, and saves
		the result in a file 'network_output.txt' in the same form as input. Can be used by latgen-faster-mapped (kaldi).

		Args:
			feats_file:		File with utterance-IDs and feature vectors.

			SNR:			SNR to filter by. Expects SNR to be stated in utterance ID.
				default - None


		'''

		if SNR == None:
			dt_file = feats_file
		else:
			subprocess.call("sed -n '/.*" + SNR + ".*/,/]$/p' " + feats_file + ">temp.txt", shell=True)
			dt_file = "temp.txt"

		if SNR != None:
			subprocess.call("rm temp.txt", shell=True)
		len_feat = 40
		print(len_feat)

		with open('network_output.txt', 'w+') as ofile:
			if splicing != 0:
				zeropad = np.zeros((splicing, len_feat))
				for k, v in self.feats_extractor(dt_file):
					utt_feats = np.concatenate((np.concatenate((zeropad, np.asarray(v))), zeropad))
					utt_len = len(utt_feats)
					f_mat = np.zeros((utt_len - 2 * splicing, len_feat * (2 * splicing + 1)), dtype=np.float32)

					for i in xrange(splicing, utt_len - splicing):
						if i < utt_len - splicing - 1:
							f_mat[i - splicing] = utt_feats[i - splicing:i + splicing + 1].flatten()
						else:
							f_mat[i - splicing] = utt_feats[i - splicing:].flatten()

					# Transpose to fit trailing dimensions for broadcasting (normalizing).
					f_mat = np.divide(np.subtract(f_mat.transpose(), np.mean(f_mat, axis=1)),
									  np.std(f_mat, axis=1)).transpose()
					ofile.write('{0}  [\n'.format(k))
					outs = self.forward_pass(f_mat, apply_log=True)
					for out in outs[:-1]:
						ofile.write('{0}\n'.format(' '.join(str(c) for c in out)))
					ofile.write('{0}  ]\n'.format(' '.join(str(c) for c in outs[-1])))
			else:
				for k, v in self.feats_extractor(dt_file):
					ofile.write('{0}  [\n'.format(k))
					f = np.asarray(v)
					f = np.divide(np.subtract(f.transpose(), np.mean(f, axis=1)), np.std(f, axis=1)).transpose()
					outs = self.forward_pass(f, apply_log=True)
					for out in outs[:-1]:
						ofile.write('{0}\n'.format(' '.join(str(c) for c in out)))
					ofile.write('{0}  ]\n'.format(' '.join(str(c) for c in outs[-1])))


	def save(self, name='NN_model', kp_prob=1, lam=0, score=None, log_scores=None, Epochs='N/A', load='None'):
		'''
		Saves the model (weights (vector of matrices) and bias (matrix)) to 'NN_model.ckpt' or input string.
		'''
		with open('Network_saves.txt','a') as f:
			if score:
				f.write("Shape: {0}\tFile: {1}\t 1-Dropout: {2}\t\tScores: {7} {3}\t Lambda: {4}\t Epochs: {5}\t Loaded from: {6}\n".format(self.shape, name, kp_prob, score, lam, Epochs, load, log_scores))

		save_path = self.saver.save(self.sess, "trained_networks/"+name+".ckpt")
		print("Model saved as: %s"%save_path)


	def load(self, n):
		'''
		Input: File where model is saved. Note network to be loaded must have the same shape as the saved network.
		'''
		self.saver.restore(self.sess,n)
		print("Model loaded.")

	def stop(self):
		self.sess.close()
		print("Session closed.")


def pretrain_network(shape, data_file, epochs, batch_size, eta, val_file='None', kp_prob=1, name='params'):
	'''
		Trains and saves a Network layer by layer. Training data is partially scored, as well as val data for each epoch.
		Final layer is trained for only one epoch.

	Args:
		shape:			List of layer sizes. Example: [39,512,512,40]
		data_file: 	 	hdf5 file containing feature and target datasets. Features must be in dataset 'feats', targets in dataset 'targs'.
		val_file: 		hdf5 file containing feature and target datasets. Features must be in dataset 'feats', targets in dataset 'targs'.
		batch_size:		Size of mini-batch.
		epochs: 		How many times to perform SGD through all layers.
		eta: 			Learning rate.
		kp_prob:		Fraction of neurons not to drop.
		name:			Name of file to save the weights in.

		Based on this: http://research.microsoft.com/pubs/157341/FeatureEngineeringInCD-DNN-ASRU2011-pub.pdf
		Although method not identical (optimal implementation not yet clear).
	'''

	input_layer = tf.placeholder("float32", shape=[None, shape[0]])
	targets = tf.placeholder("int64", shape=[None, ])

	pt_weights = []
	pt_biases = []
	pt_a = []
	pt_a_drop = []
	sm_out = []
	sm_out_calc = []
	cost = []
	train_opt = []
	acc = []

	df = h5py.File(data_file, 'r')
	feats, targs = df['feats'], df['targs']
	f_t, t_t = feats[:50000], targs[:50000]
	if val_file != 'None':
		vf = h5py.File(val_file, 'r')
		f_sc, t_sc = vf['feats'][:50000], vf['targs'][:50000]
	len_feat = len(f_t[0])

	sm_weights = []
	sm_bias = []

	for i, len_layer in enumerate(shape[1:-1]):
		pt_weights.append(ini_weight_var([shape[i], len_layer], 'w' + str(i)))
		pt_biases.append(ini_bias_var([len_layer], 'b' + str(i)))
		if i == 0:
			pt_a.append(tf.nn.elu(tf.matmul(input_layer, pt_weights[i]) + pt_biases[i]))
			pt_a_drop.append(tf.nn.dropout(pt_a[i], kp_prob))
		else:
			pt_a.append(tf.nn.elu(tf.matmul(pt_a_drop[i - 1], pt_weights[i]) + pt_biases[i]))
			pt_a_drop.append(tf.nn.dropout(pt_a[i], kp_prob))

		if len_layer == shape[-2]:
			sm_weights.append(ini_weight_var([len_layer, shape[-1]], name='wl'))
			sm_bias.append(ini_bias_var([shape[-1]], name='bl'))
		else:
			sm_weights.append(ini_weight_var([len_layer, shape[-1]]))
			sm_bias.append(ini_bias_var([shape[-1]]))
		sm_out.append(tf.matmul(pt_a_drop[i], sm_weights[i]) + sm_bias[i])  # Tensorflow takes care of softmax
		sm_out_calc.append(tf.nn.softmax(sm_out[i]))
		acc.append(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(sm_out_calc[i], 1), targets), "float")))
		#wght_num = np.sum([f*s for f, s in zip(shape[:i+1], shape[1:i+2])])
		cost.append(tf.nn.sparse_softmax_cross_entropy_with_logits(sm_out[i], targets)) #+
					#lam*tf.add_n([tf.nn.l2_loss(w_mat) for w_mat in pt_weights[:i+1]])/wght_num)
		train_opt.append(tf.train.AdamOptimizer(eta).minimize(cost[i]))

	param_dict = {}
	for i in xrange(len(pt_weights)+1):
		if i == len(pt_weights):
			param_dict['wl'] = sm_weights[-1]
			param_dict['bl'] = sm_bias[-1]
		else:
			param_dict['w'+str(i)] = pt_weights[i]
			param_dict['b'+str(i)] = pt_biases[i]

	save_op = tf.train.Saver(param_dict)

	div = 2
	div_batch = int(batch_size / div)
	num_iters = int(len(feats) / div_batch)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in xrange(epochs):
			for i in xrange(len(shape[1:-1])):
				for batch_f, batch_t in batch_gen(feats, targs, div_batch, num_iters, len_feat, div):

					sess.run(train_opt[i], feed_dict={input_layer: batch_f, targets: batch_t})

			print("train {0}, val {1}".format(sess.run(acc[-1], feed_dict={input_layer: f_t, targets: t_t}),
											  sess.run(acc[-1], feed_dict={input_layer: f_sc, targets: t_sc})))
		print



		save_op.save(sess, name)

	df.close()
	if val_file!='None':
		vf.close()

