import glob
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed
from tensorflow.keras.layers import Dropout, Bidirectional
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from tensorflow.keras.layers import Layer

# could use `outfiles` param as well
files = glob.glob("./ner/*.tags")

data_pd = pd.concat([pd.read_csv(f, header=None,
                                 names=["text", "label", "pos"], encoding='unicode_escape')
                     for f in files], ignore_index=True)

print(data_pd.info())

text_tok = Tokenizer(filters='[\\]^\t\n', lower=False,
                     split=' ', oov_token='<OOV>')

pos_tok = Tokenizer(filters='\t\n', lower=False,
                    split=' ', oov_token='<OOV>')

ner_tok = Tokenizer(filters='\t\n', lower=False,
                    split=' ', oov_token='<OOV>')

text_tok.fit_on_texts(data_pd['text'])
pos_tok.fit_on_texts(data_pd['pos'])
ner_tok.fit_on_texts(data_pd['label'])

ner_config = ner_tok.get_config()
text_config = text_tok.get_config()

print(ner_config)

text_vocab = eval(text_config['index_word'])
ner_vocab = eval(ner_config['index_word'])

print("Unique words in vocab:", len(text_vocab))
print("Unique NER tags in vocab:", len(ner_vocab))

x_tok = text_tok.texts_to_sequences(data_pd['text'])
y_tok = ner_tok.texts_to_sequences(data_pd['label'])

print(text_tok.sequences_to_texts([x_tok[1]]), data_pd['text'][1])
print(ner_tok.sequences_to_texts([y_tok[1]]), data_pd['label'][1])

max_len = 50

x_pad = sequence.pad_sequences(x_tok, padding='post',
                               maxlen=max_len)
y_pad = sequence.pad_sequences(y_tok, padding='post',
                               maxlen=max_len)

print(x_pad.shape, y_pad.shape)

print(text_tok.sequences_to_texts([x_pad[1]]))

print(ner_tok.sequences_to_texts([y_pad[1]]))

num_classes = len(ner_vocab) + 1

Y = tf.keras.utils.to_categorical(y_pad, num_classes=num_classes)
print(Y.shape)

# Length of the vocabulary
vocab_size = len(text_vocab) + 1

# The embedding dimension
embedding_dim = 64

# Number of RNN units
rnn_units = 100

# batch size
BATCH_SIZE = 90

# num of NER classes
num_classes = len(ner_vocab) + 1

dropout = 0.2

X = x_pad

# create training and testing splits
total_sentences = 62010
test_size = round(total_sentences / BATCH_SIZE * 0.2)
X_train = X[BATCH_SIZE * test_size:]
Y_train = Y[BATCH_SIZE * test_size:]

X_test = X[0:BATCH_SIZE * test_size]
Y_test = Y[0:BATCH_SIZE * test_size]

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)





class CRFLayer(Layer):
    """
    Computes the log likelihood during training
    Performs Viterbi decoding during prediction
    """

    def __init__(self,
                 label_size,
                 mask_id=0,
                 trans_params=None,
                 name='crf',
                 **kwargs):
        super(CRFLayer, self).__init__(name=name, **kwargs)
        self.label_size = label_size
        self.mask_id = mask_id
        self.transition_params = None

        if trans_params is None:  # not reloading pretrained params
            self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)),
                                                 trainable=False)
        else:
            self.transition_params = trans_params

    def get_seq_lengths(self, matrix):
        # matrix is of shape (batch_size, max_seq_len)
        mask = tf.not_equal(matrix, self.mask_id)
        seq_lengths = tf.math.reduce_sum(
            tf.cast(mask, dtype=tf.int32),
            axis=-1)
        return seq_lengths

    def call(self, inputs, seq_lengths, training=None):
        if training is None:
            training = K.learning_phase()

        # during training, this layer just returns the logits
        if training:
            return inputs

        # viterbi decode logic to return proper
        # results at inference
        _, max_seq_len, _ = inputs.shape
        seqlens = seq_lengths
        paths = []
        for logit, text_len in zip(inputs, seqlens):
            viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len],
                                                      self.transition_params)
            paths.append(self.pad_viterbi(viterbi_path, max_seq_len))

        return tf.convert_to_tensor(paths)

    def pad_viterbi(self, viterbi, max_seq_len):
        if len(viterbi) < max_seq_len:
            viterbi = viterbi + [self.mask_id] * (max_seq_len - len(viterbi))
        return viterbi

    def get_proper_labels(self, y_true):
        shape = y_true.shape
        if len(shape) > 2:
            return tf.argmax(y_true, -1, output_type=tf.int32)
        return y_true

    def loss(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(self.get_proper_labels(y_true), y_pred.dtype)

        seq_lengths = self.get_seq_lengths(y_true)
        log_likelihoods, self.transition_params = tfa.text.crf_log_likelihood(y_pred,
                                                                              y_true, seq_lengths)

        # save transition params
        self.transition_params = tf.Variable(self.transition_params, trainable=False)
        # calc loss
        loss = - tf.reduce_mean(log_likelihoods)
        return loss





class NerModel(tf.keras.Model):
    def __init__(self, hidden_num, vocab_size, label_size, embedding_size,
                 name='BilstmCrfModel', **kwargs):
        super(NerModel, self).__init__(name=name, **kwargs)
        self.num_hidden = hidden_num
        self.vocab_size = vocab_size
        self.label_size = label_size

        self.embedding = Embedding(vocab_size, embedding_size,
                                   mask_zero=True, name="embedding")
        self.biLSTM = Bidirectional(LSTM(hidden_num, return_sequences=True), name="bilstm")
        self.dense = TimeDistributed(tf.keras.layers.Dense(label_size), name="dense")
        self.crf = CRFLayer(self.label_size, name="crf")

    def call(self, text, labels=None, training=None):
        seq_lengths = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0),
                                                 dtype=tf.int32), axis=-1)

        if training is None:
            training = K.learning_phase()

        inputs = self.embedding(text)
        bilstm = self.biLSTM(inputs)
        logits = self.dense(bilstm)
        outputs = self.crf(logits, seq_lengths, training)

        return outputs


# Length of the vocabulary in chars
vocab_size = len(text_vocab)+1 # len(chars)

# The embedding dimension
embedding_dim = 64

# Number of RNN units
rnn_units = 100

#batch size
BATCH_SIZE=90

# num of NER classes
num_classes = len(ner_vocab)+1

blc_model = NerModel(rnn_units, vocab_size, num_classes, embedding_dim, dynamic=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


# create training and testing splits
total_sentences = 62010
test_size = round(total_sentences / BATCH_SIZE * 0.2)
X_train = x_pad[BATCH_SIZE*test_size:]
Y_train = Y[BATCH_SIZE*test_size:]

X_test = x_pad[0:BATCH_SIZE*test_size]
Y_test = Y[0:BATCH_SIZE*test_size]
Y_train_int = tf.cast(Y_train, dtype=tf.int32)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train_int))
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

loss_metric = tf.keras.metrics.Mean()

epochs = 5

# Iterate over epochs.
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (text_batch, labels_batch) in enumerate(train_dataset):
        labels_max = tf.argmax(labels_batch, -1, output_type=tf.int32)
        with tf.GradientTape() as tape:
            logits = blc_model(text_batch, training=True)
            loss = blc_model.crf.loss(labels_max, logits)

            grads = tape.gradient(loss, blc_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, blc_model.trainable_weights))

            loss_metric(loss)
        if step % 50 == 0:
            print('step %s: mean loss = %s' % (step, loss_metric.result()))


Y_test_int = tf.cast(Y_test, dtype=tf.int32)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test_int))
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)


out = blc_model.predict(test_dataset.take(1))


# check the outputs
print(out[1], tf.argmax(Y_test[1], -1))
print(out[2], tf.argmax(Y_test[2], -1))


text_tok.sequences_to_texts([X_test[2]])


print("Ground Truth: ", ner_tok.sequences_to_texts([tf.argmax(Y_test[2], -1).numpy()]))
print("Prediction: ", ner_tok.sequences_to_texts([out[2]]))


print(ner_tok.sequences_to_texts([tf.argmax(Y_test[1], -1).numpy()]))
print(ner_tok.sequences_to_texts([out[1]]))

blc_model.summary()


def np_precision(pred, true):
    # expect numpy arrays
    assert pred.shape == true.shape
    assert len(pred.shape) == 2
    mask_pred = np.ma.masked_equal(pred, 0)
    mask_true = np.ma.masked_equal(true, 0)
    acc = np.equal(mask_pred, mask_true)
    return np.mean(acc.compressed().astype(int))


print(np_precision(out, tf.argmax(Y_test[:BATCH_SIZE], -1).numpy()))
