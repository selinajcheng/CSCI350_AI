# Quiz 5: Selina Cheng, Michael Frid, Cecilia D'LCano

## Instructions to run
1. Create a virtual environment
    1. Create the environment: `python3 -m venv my_venv`
    2. Enter/activate the environment: `source my_venv/bin/activate`
2. Download packages in the environment
    - `pip install tensorflow numpy`
3. Run the file: `python Quiz\ 5/quiz_5.py`
4. When you're done, exit the environment: `deactivate`

## Questions and Answers
- Project choice and motivation
– Architecture explanation (RNN/LSTM structure)
– How the sequence model works in your context
– Results and observations
– Team member contributions


1. Project Choice and Motivation:

We chose to do the Text Generation with RNN/LSTM project for a number of reasons. Primarily, we as a group were interested in the challenge of text generation, and having never done anything regarding it previously were excited to try. Some of our most frequent interactions with technology involve generated text, and getting a taste of how text generation worked was an alluring prospect.

2. Architecture Explanation (LSTM Structure):

RNN model uses the previous states to predict the next state. LSTM can problem-solve long-term dependecies which is used to preserve information. In the context of the assignment, the LSTM model’s goal is to use a language model that will take Shakespeare’s text as an input and predict the next character in the sequence. This is done by inputting a word and coverting the input into a vector which is then encoded into a hidden state. This hidden state is decoded for the next input and added to the next state when the new input is encoded. While this is happenning, the hidden states are evaulated to determine if it is data worth remembering, this happens during the forget gate. If the data is sent to be forgotton, it is sent to the cell gate. The cell gate determines how much of the data will be forgotten and transforms it into new memory, which is added to the input to form a new hidden cell. This entire process compresses the inputs and adds a weight to them. This weight essentially provides context for the LSTM model to help in the prediction process. 

3. How the Sequence Model Works in Your Context:

Our LSTM model has 512 hidden layers that controls the size and capacity of the model. It uses categorical crossentryopy as the loss model. The model trains from a sample of the Shakespear data, learning how to predict the next sequence of words in a sentence and eventually mimicing Shakespeare’s style of writing. The input is text from the Shakespeare text file, it is formatted as word IDs; each input does not exceed a specific length, which would be the largest word in the file. If the input is too small, it is padded with zeroes while the longer inputs are trimmed. In each iteration the model receives the input data and before each input is encoded the model evaluates the relevance of the previous data, essentially the weight, of each of the previous words. The weights influence the prediction of the next word. If the previous words mention a character or object, its relevance would remain high in order to predict the next word. In comparison, a filler word, like an adjective, is not as relavent to the predicted word unless the adjective was the previous input. This influence of a previous word is deterimined in the forget state. If the data is determined to be irrelevant, the LSTM model mathematically adjusts its cell state, which stores long-term information, and its hidden state, which stores short-term information. The LSTM model learns which words should remain influential over time and which lose relevance as the sentence continues.

4. Results and Observations:

We ran into trouble with the model, and have not been able to see results in action. Unfortunately, we were unable to observe whether or not our model was successful.

5. Team Member Contributions:

All 3 of us worked on the project together, starting it in class and them continuing to work at home as we hadn't finished in the allotted classtime. We then gathered together virtually to write the README and summarize our work.