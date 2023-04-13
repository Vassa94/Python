import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.utils import pad_sequences

df = pd.read_csv('preguntas.csv')

preguntas = df['Pregunta']
respuestas = df['Respuesta']


tokenizer = Tokenizer()
tokenizer.fit_on_texts(preguntas)

preguntas_seq = tokenizer.texts_to_sequences(preguntas)
respuestas_seq = tokenizer.texts_to_sequences(respuestas)

MAX_SEQUENCE_LENGTH = 100
num_words = len(tokenizer.word_index)
EMBEDDING_DIM = 100

preguntas_seq = pad_sequences(preguntas_seq, maxlen=MAX_SEQUENCE_LENGTH)

model = Sequential()
model.add(Embedding(input_dim=num_words+1, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=100, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

respuestas_seq = pad_sequences(respuestas_seq, maxlen=100)
respuestas_seq = tf.keras.utils.to_categorical(respuestas_seq, num_classes=num_words+1)

model.fit(preguntas_seq, respuestas_seq, epochs=10, batch_size=32, validation_split=0.2)
