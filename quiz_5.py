import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Dense , Embedding
import numpy as np

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# 1. Prepare text data
# Shakespeare dataset within: https://www.tensorflow.org/text/tutorials/text_generation
text = open(path_to_file, 'rb').read().decode(encoding='utf-8') # Load your text corpus
print(f'Length of text: {len(text)} characters')
# Tokenize and create sequences
vocab = sorted(set(text))
chars = tf.strings.unicode_split(text, input_encoding='UTF-8')

char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

examples_per_epoch = len(text) // (sequence_length + 1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(sequence_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# 2. Build LSTM model
vocab_size = len(vocab) # of unique_words
embedding_dim = 256
lstm_units = 512
sequence_length = 100


model = Sequential ([
    Embedding ( vocab_size , embedding_dim ,
    input_length = sequence_length ) ,
    LSTM ( lstm_units , return_sequences = True ) ,
    LSTM ( lstm_units ) ,
    Dense ( vocab_size , activation ='softmax')
])

# 3. Train model 
model.compile ( loss = 'categorical_crossentropy',
                optimizer = 'adam')

# 4. Generate text
def generate_text ( seed_text , length =100 , temperature =1.0) :
    input_ids = [char2idx[c] for c in seed_text]
    input_ids = tf.expand_dims(input_ids, 0)

    generated_chars = []

    for i in range(length):
        predictions = model.predict(input_ids, verbose=0)
        predictions = predictions[:, -1, :]    
        predictions = predictions / temperature 
        probs = tf.nn.softmax(predictions).numpy().ravel()

        next_id = np.random.choice(vocab_size, p=probs)
        next_char = idx2char[next_id]

        generated_chars.append(next_char)

        input_ids = tf.concat(
            [input_ids, tf.expand_dims([next_id], 0)],
            axis=1
        )

        input_ids = input_ids[:, -sequence_length:]

    return seed_text + ''.join(generated_chars)

print("\nGenerating text...")
seed = "ROMEO: "
generated = generate_text(seed, length=200, temperature=1.0)
print(generated)