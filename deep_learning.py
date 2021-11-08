import random as python_random
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import warnings
from tensorflow import keras
from keras.layers import LSTM, Activation, Dropout, Dense, Input, CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
import transformers
from transformers import (OpenAIGPTTokenizer, TFOpenAIGPTForSequenceClassification, MobileBertTokenizer,
                          TFMobileBertForSequenceClassification, TFAutoModelForSequenceClassification,
                          AutoTokenizer, BertTokenizerFast, TFBertForSequenceClassification,
                          DistilBertTokenizer, TFDistilBertForSequenceClassification,
                          RobertaTokenizer, TFRobertaForSequenceClassification,
                          XLNetTokenizer, TFXLNetForSequenceClassification)

warnings.filterwarnings("ignore")


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-lstm", "--lstm", action="store_true",
                        help="Use the LSTM for classification")

    parser.add_argument("-ct", "--custom_test_set", action="store_true",
                        help="Use custom test set to test model")

    parser.add_argument("-val", "--val_set", action="store_true",
                        help="Use val set to test model")

    parser.add_argument("-epoch", "--epoch_size", default=10, type=float,
                        help="Number of epochs to train")

    parser.add_argument("-batch", "--batch_size", default=16, type=float,
                        help="Number of epochs to train")

    parser.add_argument("-s", "--seed", default=42, type=int,
                        help="Seed for model trainings (default 42)")

    parser.add_argument("-bert_pretrained", "--bert_pretrained", action="store_true",
                        help="Use pretrained BERT for classification")

    parser.add_argument("-lstm_pretrained", "--lstm_pretrained", action="store_true",
                        help="Use pretrained LSTM for classification")

    args = parser.parse_args()
    return args


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    python_random.seed(seed)


def test_set_predict(model, X_test, Y_test, ident):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)

    print('Accuracy on {1} set: {0}'.format(round(accuracy_score(Y_test, Y_pred), 3), ident))


def read_glove_vector(glove_vec):
    embeddings_index = {}
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    return embeddings_index


def bert_model(X_train, X_dev):
    lm = "bert-base-uncased"
    optim = Adam(learning_rate=5e-5)
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    tokenizer = BertTokenizerFast.from_pretrained(lm)
    model = TFBertForSequenceClassification.from_pretrained(lm, num_labels=9)
    tokenizer.pad_token = "[PAD]"
    tokens_train = tokenizer(X_train.values.tolist(), padding=True, max_length=256, truncation=True,
                             return_tensors="np").data
    tokens_dev = tokenizer(X_dev.values.tolist(), padding=True, max_length=256, truncation=True,
                           return_tensors="np").data
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])

    return model, tokenizer, tokens_train, tokens_dev


def emb_matrix(word_to_vec_map, words_to_index, maxLen):
    vocab_len = len(words_to_index)
    embed_vector_len = word_to_vec_map['moon'].shape[0]

    emb_matrix = np.zeros((vocab_len, embed_vector_len))

    for word, index in words_to_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            emb_matrix[index, :] = embedding_vector

    embedding_layer = Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen,
                                weights=[emb_matrix], trainable=False)

    return embedding_layer


def lstm_model(input_shape, embedding_layer):
    adam = Adam(learning_rate=0.0005)

    X_indices = Input(input_shape)

    embeddings = embedding_layer(X_indices)

    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.2)(X)
    X = LSTM(128, return_sequences=True)(X)
    X = Dropout(0.2)(X)
    X = LSTM(128, return_sequences=True)(X)
    X = keras.layers.BatchNormalization()(X)
    X = Dropout(0.2)(X)
    X = LSTM(128)(X)
    X = Dropout(0.2)(X)
    X = Dense(9, activation='softmax')(X)

    model = Model(inputs=X_indices, outputs=X)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train(model, X_train, Y_train_bin, X_test, Y_test_bin, epochs, batch_size, filename, custom_test_set, val_set):
    verbose = 1
    batch_size = batch_size
    epochs = epochs
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights='True', patience=6)
    cp = tf.keras.callbacks.ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit(X_train, Y_train_bin,
              verbose=verbose,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[callback, cp],
              validation_data=(X_test, Y_test_bin))

    if custom_test_set:
        test_set_predict(model, X_test, Y_test_bin, "custom_test")
    else:
        if val_set:
            test_set_predict(model, X_test, Y_test_bin, "val")
        else:
            test_set_predict(model, X_test, Y_test_bin, "test")

    return model


def main():
    args = create_arg_parser()
    set_seed(args.seed)

    train_df = pd.read_csv('./processed_data/processed_train.csv')

    if args.custom_test_set:
        test_df = pd.read_csv('./processed_data/processed_custom_test.csv')
    else:
        if args.val_set:
            test_df = pd.read_csv('./processed_data/processed_val.csv')
        else:
            test_df = pd.read_csv('./processed_data/processed_test.csv')

    if args.lstm:
        X_train, Y_train = train_df['clean'], train_df['newspaper_name']
        X_test, Y_test = test_df['clean'], test_df['newspaper_name']
    else:
        X_train, Y_train = train_df['body'], train_df['newspaper_name']
        X_test, Y_test = test_df['body'], test_df['newspaper_name']

    encoder = LabelBinarizer()
    encode = encoder.fit(Y_train.tolist())
    Y_train_bin = encode.transform(Y_train.tolist())
    # Use encoder.classes_ to find mapping back
    Y_test_bin = encode.transform(Y_test.tolist())

    if args.lstm:
        filename = "./model/lstm.h5"
        if args.lstm_pretrained:
            with open(filename, 'rb') as file:
                model = tf.keras.models.load_model(file)
            if args.custom_test_set:
                test_set_predict(model, X_test, Y_test_bin, "custom_test")
            else:
                if args.val_set:
                    test_set_predict(model, X_test, Y_test_bin, "val")
                else:
                    test_set_predict(model, X_test, Y_test_bin, "test")
        else:
            tokenizer = Tokenizer(num_words=5000)
            tokenizer.fit_on_texts(X_train)

            words_to_index = tokenizer.word_index
            word_to_vec_map = read_glove_vector('glove.6B.50d.txt')

            maxLen = 300

            embedding_layer = emb_matrix(word_to_vec_map, words_to_index, maxLen)

            X_train_indices = tokenizer.texts_to_sequences(X_train)
            X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')

            X_test_indices = tokenizer.texts_to_sequences(X_test)
            X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen, padding='post')

            model = lstm_model(maxLen, embedding_layer)
            model = train(model, X_train_indices, Y_train_bin, X_test_indices, Y_test_bin, args.epoch_size,
                          args.batch_size, filename, args.custom_test_set, args.val_set)
    else:
        filename = "./model/bert.h5"
        if args.bert_pretrained:
            with open(filename, 'rb') as file:
                model = tf.keras.models.load_model(file)
            if custom_test_set:
                test_set_predict(model, X_test, Y_test_bin, "custom_test")
            else:
                if val_set:
                    test_set_predict(model, X_test, Y_test_bin, "val")
                else:
                    test_set_predict(model, X_test, Y_test_bin, "test")
        else:
            model, tokenizer, tokens_train, tokens_test = bert_model(X_train, X_test)
            model = train(model, tokens_train, Y_train_bin, tokens_test, Y_test_bin, args.epoch_size, args.batch_size, filename, args.custom_test_set, args.val_set)

            
if __name__ == '__main__':
    main()
