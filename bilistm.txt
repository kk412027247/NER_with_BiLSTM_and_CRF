import glob
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense

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


def build_model_bilstm(vocab_size, embedding_dim, rnn_units, batch_size, classes):
    model = tf.keras.Sequential([
        Embedding(vocab_size, embedding_dim, mask_zero=True,
                  batch_input_shape=[batch_size, None]),
        Bidirectional(LSTM(units=rnn_units,
                           return_sequences=True,
                           dropout=dropout,
                           kernel_initializer=tf.keras.initializers.he_normal())),
        TimeDistributed(Dense(rnn_units, activation='relu')),
        Dense(num_classes, activation="softmax")
    ])

    return model


model = build_model_bilstm(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE,
    classes=num_classes)
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

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

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=15)

# batch size in eval
model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)

y_pred = model.predict(X_test, batch_size=BATCH_SIZE)

print(text_tok.sequences_to_texts([X_test[1]]))

print(ner_tok.sequences_to_texts([y_pad[1]]))

y_pred = tf.argmax(y_pred, -1)

print(y_pred.shape)

y_pnp = y_pred.numpy()

print(ner_tok.sequences_to_texts([y_pnp[1]]))
