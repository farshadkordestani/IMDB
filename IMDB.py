# 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative).
# Reviews have been preprocessed, and each review is encoded as a sequence of word indexes
# words are indexed by overall frequency, so that for instance the integer "3" encodes the 3rd most frequent
from pandas import read_csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, SimpleRNN
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt


num_words, max_len, embed_len = 1024, 256, 128
data = read_csv('imdb.csv')
x_train, x_test, y_train, y_test = train_test_split(data['review'].to_numpy(),
                                                    data['sentiment'].replace({'negative': 0, 'positive': 1}).to_numpy(),
                                                    test_size=0.3)
t = Tokenizer(num_words)
t.fit_on_texts(x_train)
encoded_x_train = t.texts_to_sequences(x_train)
encoded_x_test = t.texts_to_sequences(x_test)
x_train = pad_sequences(encoded_x_train, maxlen=max_len)
x_test = pad_sequences(encoded_x_test, maxlen=max_len)
model = Sequential()
model.add(Embedding(num_words, output_dim=embed_len, input_length=max_len))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_split=0.1)
print(model.summary())
print(model.evaluate(x_test, y_test))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(history.epoch, history.history['accuracy'], label='Train accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.show()
