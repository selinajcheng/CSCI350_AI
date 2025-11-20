# Referencing https://www.tensorflow.org/text/tutorials/text_generation 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Dense , Embedding
import numpy as np
import random

# Random seeding for reproducibility (isolated/less confounded testing)
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# 1. Prepare text data
# Download and read data
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print(f'Length of text: {len(text)} characters')    # quick check

# Tokenize and create sequences
vocab = sorted(set(text))   # vocab contains all unique characters from our text
chars = tf.strings.unicode_split(text, input_encoding='UTF-8')

# Map from char to num and num to char to allow NNs (neural nets) to process
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

# Convert all text to num
text_as_int = np.array([char2idx[c] for c in text])

sequence_length = 100   # the model will see 100 characters in each sequence to predict the next
examples_per_epoch = len(text) // (sequence_length + 1)

# Create text sequences (of 100 characters) to use in training
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(sequence_length + 1, drop_remainder=True)

# function that will be used to give the model the input (first chunk of text) and train it to predict the target (second chunk of text)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# Apply the function to all the sequences we made earlier
dataset = sequences.map(split_input_target)

# Configure dataset for training
BATCH_SIZE = 64         # process the dataset in smaller chunks (batches). in this case: 64 sequences at a time
BUFFER_SIZE = 10000     # shuffle the dataset 10,000 sequences at a time
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# 2. Build LSTM model
vocab_size = len(vocab)     # number of unique characters the model knows
embedding_dim = 256         # size of vector used to represent each character
lstm_units = 512            # number of memory units

model = Sequential ([
    Embedding ( vocab_size , embedding_dim , input_length = sequence_length ) ,
    LSTM ( lstm_units , return_sequences = True ) ,
    LSTM ( lstm_units , return_sequences = True ) ,
    Dense ( vocab_size , activation = 'softmax')
])


model.compile ( loss = 'sparse_categorical_crossentropy',
                optimizer = 'adam' )
model.build(input_shape=(None, sequence_length))

print("\nLSTM model built.")
print("\nBelow is the model summary:")
print(model.summary())

# 3. Train model
history = model.fit(dataset, epochs=10, verbose=1)

print("\nModel trained.")

# 4. Generate text
def generate_text ( seed_text , length =100 , temperature =1.0) :
    input_eval = [char2idx[c] for c in seed_text]

    if len(input_eval) < sequence_length:
        padding = [0] * (sequence_length - len(input_eval))
        input_eval = padding + input_eval
    else:
        input_eval = input_eval[-sequence_length:]

    input_eval = tf.expand_dims(input_eval, 0)

    generated_chars = []

    for i in range(length):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :]    # last time step prediction

        predictions = predictions / temperature     # apply specified temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0,0].numpy()
        generated_chars.append(idx2char[predicted_id])

        input_eval = tf.concat([input_eval[:, 1:], [[predicted_id]]], axis=1)

    return seed_text + ''.join(generated_chars)

print("\nGenerating text...")
seed = "ROMEO: "
generated = generate_text(seed, length=200, temperature=1.0)
print(generated)