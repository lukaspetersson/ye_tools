# credit to: https://www.tensorflow.org/text/tutorials/text_generation

import tensorflow as tf
print(tf.__version__)
import numpy as np
import os
import time
import string


text = open("note_beats.txt", 'rb').read().decode(encoding='utf-8')

#create the alphabet
vocab = sorted(set(text))
#methods to convert id <-> char
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

#convert ids to text
def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

#create training examples/targets
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
seq_len = 100
ex_per_epoch = len(text)//(seq_len+1)
seqs = ids_dataset.batch(seq_len+1, drop_remainder = True)
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = seqs.map(split_input_target)
BATCH_SIZE = 10
BUFFER_SIZE = 10000
dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

model = MyModel(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

#Change to True for retraining
if False:
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    EPOCHS = 40
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
else:
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)


## Print to abc format

# decode the character to the number of beats in the abc format
# use 1/16 as default
def char_to_beats(c):
    alphabet = list(string.ascii_lowercase)
    if c in alphabet:
        return alphabet.index(c)
    return 0

states = None
# Pick starting notes
next_char = tf.constant(['ccbbc'])
result = [next_char]
song = "|"
num_bars = 20
for _ in range(num_bars):
    measure = []
    # loop until the meassure is filled (4/4 time signature)
    while True:
        # Use model to predict next char
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        c = tf.strings.join([next_char])[0].numpy().decode("utf-8")
        # How many beats in the meassure if the note is used
        s = sum([char_to_beats(c) for c in measure]) + char_to_beats(c) 
        if s <= 16:
            if c != "":
                measure.append(c)
        if s == 16:
            break
    for b in measure:
        song += ("*"+str(char_to_beats(b))+" ")
    song += "|"

        
f = open("generated_beats.txt", "w")
f.write(song)
f.close()


