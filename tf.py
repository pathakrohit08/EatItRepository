import re
import os
import gzip
import pandas as pd
import shutil
import numpy as np
import json
import time
from tensorflow.contrib import learn
import datetime
import logging
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)


##### This code preprocess dataset 20NewsGroup it reads all the text file contents and put them in a csv file
data_path=os.path.abspath(os.path.join(os.path.curdir, "data/raw_data"))
csv_path=os.path.abspath(os.path.join(os.path.curdir, "data"))

def load_Data():
    for filename in os.listdir(data_path):
        if filename.endswith('.txt'):
            print(f'reading contents from {filename}')
            data=open(data_path+'\\'+filename,'r',encoding='UTF8',errors='ignore').read()
            lines = re.split("\n", data)
            numline = len(lines)
            stop_counter=int(0.7*numline)

            train_lines=lines[:stop_counter]
            test_lines=lines[stop_counter:]

            with open(csv_path+'\\csv_data_train.csv', 'w') as file:
                for line in train_lines:
                    file.write(filename.replace('.txt','')+','+line)
                    file.write('\n')

            with open(csv_path+'\\csv_data_test.csv', 'w') as file:
                for line in test_lines:
                    file.write(filename.replace('.txt','')+','+line)
                    file.write('\n')

    df = pd.read_csv(csv_path+'\\csv_data_train.csv', header=None,error_bad_lines=False)
    df.rename(columns={0: 'category', 1: 'description'}, inplace=True)
    df.to_csv(csv_path+'\\csvdatatrain.csv', index=False)


    df2 = pd.read_csv(csv_path+'\\csv_data_test.csv', header=None,error_bad_lines=False)
    df2.rename(columns={0: 'category', 1: 'description'}, inplace=True)
    df2.to_csv(csv_path+'\\csvdatatest.csv', index=False)



#### RNN
import tensorflow as tf


class RNN:
    def __init__(self, sequence_length,  # the length of the input sentence
                 num_classes,  # number of output classes
                 vocab_size,  # this might be the length of the whole document (need to check)
                 embedding_size,  # the size of the word embedding in Google Word2Vec it is 300 dim
                 cell_type,  # the type of cell i.e either Basic-RNN,GRU,LSTM
                 hidden_size,  # the unrolled length of the network
                 l2_reg_lambda=0.0):  # no idea

        # setting placeholders for input and output
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_values')
        self.output_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='output_values')

        # need to understand these parameters
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        l2_loss = tf.constant(0.0)

        text_length = self._length(self.input_text)

        # Setting up network
        # after batch_size*text_length we feed the data to embedding layer
        with tf.device('/cpu:0'), tf.name_scope('text-embedding'):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                      name="GoogleWord2VecModel")
            self.sequence_vector = tf.nn.embedding_lookup(self.W_text, self.input_text)

        # this sequence vector is then fed to the RNN layer
        with tf.name_scope('rnn'):
            # get the RNN cell type (BasicRNN,LSTM,GRU)
            cell = self._get_cell(hidden_size, cell_type)

            # wrap it with droppout layer
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)

            all_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=self.sequence_vector,
                                               sequence_length=text_length,
                                               dtype=tf.float32)

            # get the last timestep output
            self.h_outputs = self.last_relevant(all_outputs, text_length)
            tf.summary.histogram('rnn_out', self.h_outputs)

        # once we get the output from RNN we add weights and biases to it
        with tf.name_scope('output'):
            W = tf.get_variable('W', shape=[hidden_size, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_outputs, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # network is complete

        # calculate loss and acccuracy
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.output_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.output_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

        with tf.name_scope('num_correct'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.output_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')

    @staticmethod
    def _get_cell(hidden_size, cell_type):
        if cell_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)


def train_rnn():
    print('training on RNN')

    # loading data in memory (bad choice)
    train_file = './data/csv_data.csv.zip'
    x_raw, y_raw, df, labels = load_data_and_labels(train_file)

    parameter_file = './parameters.json'
    params = json.loads(open(parameter_file).read())

    """Step 1: pad each sentence to the same length and map each word to an id"""
    max_document_length = max([len(x.split(' ')) for x in x_raw])
    logging.info('The maximum length of all sentences: {}'.format(max_document_length))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_raw)))
    y = np.array(y_raw)

    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print("")

    """Step 2: split the original dataset into train and test sets"""
    x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    """Step 3: shuffle the train set and split the train set into train and dev sets"""
    shuffle_indices = np.random.permutation(np.arange(len(y_)))
    x_shuffled = x_[shuffle_indices]
    y_shuffled = y_[shuffle_indices]
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)

    """Step 4: save the labels into labels.json since predict.py needs it"""
    with open('./labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4)

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))


    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            rnn = RNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=params['embedding_dim'],
                cell_type='vanilla',
                hidden_size=128,
                l2_reg_lambda=params['l2_reg_lambda'])

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_rnn", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)
            weights_summary = tf.summary.histogram("weights", rnn.h_outputs)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, weights_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver()

            # One training step: train the model with one batch
            def train_step(x_batch, y_batch):
                feed_dict = {
                    rnn.input_text: x_batch,
                    rnn.output_y: y_batch,
                    rnn.dropout_keep_prob: params['dropout_keep_prob']}
                _, step,summaries, loss, acc = sess.run([train_op, global_step,train_summary_op, rnn.loss, rnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, acc))
                train_summary_writer.add_summary(summaries, step)
            #print("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, acc))

            # One evaluation step: evaluate the model with one batch
            def dev_step(x_batch, y_batch):
                feed_dict = {
                    rnn.input_text: x_batch,
                    rnn.output_y: y_batch,
                    rnn.dropout_keep_prob: 1.0}
                summaries_dev,step, loss, acc, num_correct = sess.run([dev_summary_op,global_step, rnn.loss, rnn.accuracy, rnn.num_correct], feed_dict)
                dev_summary_writer.add_summary(summaries_dev, step)
                return num_correct

            # Save the word_to_id map since predict.py needs it
            vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
            best_accuracy, best_at_step = 0, 0

            """Step 6: train the cnn model with x_train and y_train (batch by batch)"""
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)

                """Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
                if current_step % params['evaluate_every'] == 0:
                    dev_batches = batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        num_dev_correct = dev_step(x_dev_batch, y_dev_batch)
                        total_dev_correct += num_dev_correct

                    dev_accuracy = float(total_dev_correct) / len(y_dev)
                    logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))

                    """Step 6.2: save the model if it is the best based on accuracy on dev set"""
                    if dev_accuracy >= best_accuracy:
                        best_accuracy, best_at_step = dev_accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.critical('Saved model at {} at step {}'.format(path, best_at_step))
                        logging.critical('Best accuracy is {} at step {}'.format(best_accuracy, best_at_step))

            """Step 7: predict x_test (batch by batch)"""
            test_batches = batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
            total_test_correct = 0
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                num_test_correct = dev_step(x_test_batch, y_test_batch)
                total_test_correct += num_test_correct

            test_accuracy = float(total_test_correct) / len(y_test)
            logging.critical('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))
            logging.critical('The training is complete')
            
            
            
 ########### helper functions
 
def load_data_and_labels(filename):
	"""Load sentences and labels"""
	#df = pd.read_csv(filename, compression='zip', dtype={'consumer_complaint_narrative': object})
	df = pd.read_csv(filename, compression='gzip',error_bad_lines=False)
	selected = ['category', 'description']
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1) # Drop non selected columns
	df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
	df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe

	# Map the actual labels to one hot labels
	labels = sorted(list(set(df[selected[0]].tolist())))
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	x_raw = df[selected[1]].apply(lambda x: (x)).tolist()
	y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
	return x_raw, y_raw, df, labels

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""Iterate the data batch by batch"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]


