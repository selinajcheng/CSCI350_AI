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
