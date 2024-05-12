import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import pandas as pd

X_MAXLEN = 104
TRUNCATING = 'post'
PADDING = 'post'

def predict_NER(sentence):
    x_seq = x_tokenizer.texts_to_sequences([sentence])
    x_pad = pad_sequences(x_seq, maxlen=X_MAXLEN, truncating=TRUNCATING, padding=PADDING)
    y_pred = model.predict(x_pad)
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = y_tokenizer.sequences_to_texts(y_pred)
    split_sentence = sentence.split()
    y_pred = y_pred[0].split()
    prediciton = []
    print('Length of sentence:', len(split_sentence))
    for i in range(len(split_sentence)):
        if y_pred[i] != 'O':
            prediciton.append((y_pred[i]))
        else:
            prediciton.append((split_sentence[i]))
    return ' '.join(prediciton)

if __name__ == '__main__':
    model = tf.keras.models.load_model('ner_model.h5')
    with open('x_tokenizer.pickle', 'rb') as handle:
        x_tokenizer = pickle.load(handle)

    with open('y_tokenizer.pickle', 'rb') as handle:
        y_tokenizer = pickle.load(handle)

    sentence = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California, that designs, develops, and sells consumer electronics, computer software, and online services. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976."
    predicted_sentence = predict_NER(sentence)
    print(sentence)
    print(predicted_sentence)



