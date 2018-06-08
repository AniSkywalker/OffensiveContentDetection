import os
import sys

from keras import Input, Model
from keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization

sys.path.append('../')
import collections
import time
import numpy
from sklearn import metrics
from keras import regularizers, initializers, constraints
from keras.models import Sequential, model_from_json
from keras.layers.core import Dropout, Dense, Activation, Reshape, Flatten, Layer
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.merge import add, concatenate, subtract
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from collections import defaultdict
import src.data_processing.data_handler as dh
import src.AttentionLayer

from keras import backend as K

numpy.random.seed(1337)





class offensive_content_model():
    _train_file = None
    _test_file = None
    _tweet_file = None
    _output_file = None
    _model_file = None
    _word_file_path = None
    _vocab_file_path = None
    _model_filename = 'model.json'
    _vocab = None
    _line_maxlen = None

    def __init__(self):
        self._line_maxlen = 50
        self._line_char_maxlen = 200

    def _build_network(self, vocab_size, char_vocab_size, emb_weights=None, hidden_units=256, trainable=False,
                       batch_size=1):
        print('Build model...')

        text_input = Input(name='text', shape=(self._line_maxlen,))

        if (len(emb_weights) == 0):
            emb = Embedding(vocab_size, 128, input_length=self._line_maxlen, embeddings_initializer='glorot_normal',
                            trainable=trainable)(text_input)
        else:
            emb = Embedding(vocab_size, emb_weights.shape[1], input_length=self._line_maxlen, weights=[emb_weights],
                            trainable=trainable)(text_input)

        emb = Reshape((int(emb.shape[1]), int(emb.shape[2]), 1))(emb)

        t_cnn1 = Convolution2D(int(hidden_units / 8), (2, 1), kernel_initializer='he_normal',
                               bias_initializer='he_normal',
                               activation='relu', padding='valid', use_bias=True, input_shape=(1, self._line_maxlen))(
            emb)
        t_cnn1 = MaxPooling2D((2, 1))(t_cnn1)
        t_cnn1 = Dropout(0.5)(t_cnn1)

        t_cnn2 = Convolution2D(int(hidden_units / 4), (2, 1), kernel_initializer='he_normal',
                               bias_initializer='he_normal',
                               activation='relu', padding='valid', use_bias=True)(t_cnn1)
        t_cnn2 = MaxPooling2D((2, 1))(t_cnn2)
        t_cnn2 = Dropout(0.5)(t_cnn2)

        t_cnn3 = Convolution2D(int(hidden_units / 4), (2, 1), kernel_initializer='he_normal',
                               bias_initializer='he_normal',
                               activation='relu', padding='valid', use_bias=True)(t_cnn2)
        t_cnn3 = MaxPooling2D((2, 1))(t_cnn3)
        t_cnn3 = Dropout(0.5)(t_cnn3)

        gmp = GlobalMaxPooling2D()(t_cnn3)

        char_input = Input(name='char_text', shape=(self._line_char_maxlen,))

        char_emb = Embedding(char_vocab_size, 25, input_length=self._line_char_maxlen,
                             embeddings_initializer='glorot_normal',
                             trainable=True)(char_input)

        char_emb = Reshape((int(char_emb.shape[1]), int(char_emb.shape[2]), 1))(char_emb)

        char_cnn1 = Convolution2D(int(hidden_units / 8), (2, 1), kernel_initializer='he_normal',
                                  bias_initializer='he_normal',
                                  activation='relu', padding='valid', use_bias=True,
                                  input_shape=(1, self._line_char_maxlen))(char_emb)
        char_cnn1 = MaxPooling2D((2, 1))(char_cnn1)
        char_cnn1 = Dropout(0.5)(char_cnn1)

        char_cnn2 = Convolution2D(int(hidden_units / 4), (2, 1), kernel_initializer='he_normal',
                                  bias_initializer='he_normal',
                                  activation='relu', padding='valid', use_bias=True)(char_cnn1)
        char_cnn2 = MaxPooling2D((2, 1))(char_cnn2)
        char_cnn2 = Dropout(0.5)(char_cnn2)

        char_cnn3 = Convolution2D(int(hidden_units / 4), (2, 1), kernel_initializer='he_normal',
                                  bias_initializer='he_normal',
                                  activation='relu', padding='valid', use_bias=True)(char_cnn2)
        char_cnn3 = MaxPooling2D((2, 1))(char_cnn3)
        char_cnn3 = Dropout(0.5)(char_cnn3)

        char_gmp = GlobalMaxPooling2D()(char_cnn3)

        merged = concatenate([char_gmp, gmp])

        output = Dense(2, activation='softmax')(merged)

        model = Model(inputs=[text_input, char_input], outputs=output)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('No of parameter:', model.count_params())

        print(model.summary())
        return model

    def _build_emotion_network(self, vocab_size, maxlen, emb_weights=None, hidden_units=256, trainable=False):
        print('Build model...')
        model = Sequential()
        # if (emb_weights == None):
        # model.add(Embedding(vocab_size, 128, input_length=maxlen, embeddings_initializer='glorot_normal'))
        # else:
        model.add(Embedding(vocab_size, emb_weights.shape[1], input_length=maxlen, weights=[emb_weights],
                            trainable=trainable))
        print(model.output_shape)

        model.add(Reshape((model.output_shape[1], model.output_shape[2], 1)))
        model.add(BatchNormalization(momentum=0.9))

        print(model.output_shape)

        model.add(Convolution2D(64, (3, 5), kernel_initializer='he_normal', padding='valid', activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.5))

        model.add(Convolution2D(128, (3, 5), kernel_initializer='he_normal', padding='valid', activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.5))

        # model.add(LSTM(hidden_units, kernel_initializer='he_normal', activation='sigmoid', return_sequences=True))
        # model.add(Dropout(0.25))
        # model.add(LSTM(hidden_units, kernel_initializer='he_normal', activation='sigmoid'))
        # model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(int(hidden_units) / 2, kernel_initializer='he_normal', activation='relu'))
        model.add(BatchNormalization(momentum=0.9))

        model.add(Dense(6))
        model.add(Activation('softmax'))
        adam = Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        print('No of parameter:', model.count_params())
        print(model.summary())
        return model


class train_model(offensive_content_model):
    train = None
    validation = None
    print("Loading resource...")

    def __init__(self, train_file, validation_file, word_file_path, split_word_path, emoji_file_path, model_file,
                 vocab_file,
                 output_file,
                 model_filename=None):
        offensive_content_model.__init__(self)

        self._train_file = train_file
        self._validation_file = validation_file
        self._word_file_path = word_file_path
        self._split_word_file_path = split_word_path
        self._emoji_file_path = emoji_file_path
        self._model_file = model_file
        self._vocab_file_path = vocab_file
        self._output_file = output_file
        self._model_filename = model_filename

        self.load_train_validation_data(lowercase=False, at_character=True)
        self.char_train = self.train
        self.char_validation = self.validation

        self.load_train_validation_data()

        # batch size
        batch_size = 16
        print('bb', len(self.train))
        print('bb', len(self.char_train))

        self.train = self.train[-len(self.train) % batch_size:]
        self.char_train = self.char_train[-len(self.char_train) % batch_size:]
        print('bb', len(self.train))
        print('bb', len(self.char_train))

        print(self._line_maxlen)
        print(self._line_char_maxlen)

        # build vocabulary
        self._vocab = dh.build_vocab(self.train)
        self._vocab['unk'] = len(self._vocab.keys()) + 1

        self._char_vocab = dh.build_vocab(self.char_train)
        self._char_vocab['unk'] = len(self._char_vocab.keys()) + 1

        print(len(self._vocab.keys()) + 1)
        print('unk::', self._vocab['unk'])
        print(len(self._char_vocab.keys()) + 1)
        print('unk::', self._char_vocab['unk'])

        dh.write_vocab(self._vocab_file_path, self._vocab)
        dh.write_vocab(self._vocab_file_path + '.char', self._char_vocab)

        # prepares input
        X, Y, D, C, A = dh.vectorize_word_dimension(self.train, self._vocab)
        X = dh.pad_sequence_1d(X, maxlen=self._line_maxlen)

        # prepares input
        tX, tY, tD, tC, tA = dh.vectorize_word_dimension(self.validation, self._vocab)
        tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)

        # prepares character input
        cX, cY, cD, cC, cA = dh.vectorize_word_dimension(self.char_train, self._char_vocab)
        cX = dh.pad_sequence_1d(cX, maxlen=self._line_char_maxlen)

        # prepares character input
        ctX, ctY, ctD, ctC, ctA = dh.vectorize_word_dimension(self.char_validation, self._char_vocab)
        ctX = dh.pad_sequence_1d(ctX, maxlen=self._line_char_maxlen)

        print("X", X.shape, cX.shape)

        # hidden units
        hidden_units = 256

        # word2vec dimension
        dimension_size = 128
        W = []
        W = dh.get_word2vec_weight(self._vocab, n=300,
                                   path='/home/striker/word2vec/GoogleNews-vectors-negative300.bin')
        # W = dh.get_glove_weights(self._vocab, n=200, path='/home/striker/word2vec/glove_model_200.txt.bin')
        print('Word2vec obtained....')

        # solving class imbalance
        ratio = self.calculate_label_ratio(Y)
        ratio = [max(ratio.values()) / value for key, value in ratio.items()]
        print('class ratio::', ratio)

        Y, tY = [np_utils.to_categorical(x) for x in (Y, tY)]
        # Y, tY = [np_utils.to_categorical(x) for x in (Y, tY)]

        print('train_X', X.shape)
        print('train_Y', Y.shape)
        print('validation_X', tX.shape)
        print('validation_Y', tY.shape)

        # trainable true if you want word2vec weights to be updated
        model = None
        if (model_filename == 'emotion.json'):
            model = self._build_emotion_network(len(self._vocab.keys()) + 1, self._line_maxlen, emb_weights=W,
                                                hidden_units=hidden_units, trainable=False)
        if (model_filename == 'offensive.json'):
            model = self._build_network(len(self._vocab.keys()) + 1, len(self._char_vocab.keys()) + 1, emb_weights=W,
                                        hidden_units=hidden_units, trainable=False, batch_size=8)

        open(self._model_file + self._model_filename, 'w').write(model.to_json())
        save_best = ModelCheckpoint(model_file + self._model_filename + '.hdf5', save_best_only=True)
        early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
        lr_tuner = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001,
                                     cooldown=0, min_lr=0.000001)

        # training
        # model.fit([X,cX], Y, batch_size=16, epochs=150, validation_split=0.2, shuffle=True,
        #           callbacks=[save_best, early_stopping, lr_tuner], class_weight=ratio, verbose=1)

        model.fit([X, cX], Y, batch_size=8, epochs=100, validation_data=([tX, ctX], tY), shuffle=True,
                  callbacks=[save_best, early_stopping, lr_tuner], class_weight=ratio)

    def load_train_validation_data(self, ignore_profiles=True, lowercase=True, at_character=False):
        self.train = dh.loaddata(self._train_file, self._word_file_path, self._split_word_file_path,
                                 self._emoji_file_path, normalize_text=True,
                                 split_hashtag=True, lowercase=lowercase,
                                 ignore_profiles=ignore_profiles, at_character=at_character)

        self.validation = dh.loaddata(self._validation_file, self._word_file_path, self._split_word_file_path,
                                      self._emoji_file_path,
                                      normalize_text=True,
                                      split_hashtag=True, lowercase=lowercase,
                                      ignore_profiles=ignore_profiles, at_character=at_character)

    def get_maxlen(self):
        return max(map(len, (x for _, x in self.train + self.validation)))

    def write_vocab(self):
        with open(self._vocab_file_path, 'w') as fw:
            for key, value in self._vocab.iteritems():
                fw.write(str(key) + '\t' + str(value) + '\n')

    def calculate_label_ratio(self, labels):
        return collections.Counter(labels)


class test_model(offensive_content_model):
    test = None
    model = None

    def __init__(self, model_file, word_file_path, split_word_path, emoji_file_path, vocab_file_path, output_file,
                 input_weight_file_path=None):
        print('initializing...')
        offensive_content_model.__init__(self)

        self._model_file = model_file
        self._word_file_path = word_file_path
        self._split_word_file_path = split_word_path
        self._emoji_file_path = emoji_file_path
        self._vocab_file_path = vocab_file_path
        self._output_file = output_file
        self._input_weight_file_path = input_weight_file_path

        print('test_maxlen', self._line_maxlen)

    def load_trained_model(self, model_file_name='.hdf5'):
        start = time.time()
        self.__load_model(self._model_file + model_file_name, self._model_file + model_file_name + '.hdf5')
        end = time.time()
        print('model loading time::', (end - start))

    def __load_model(self, model_path, model_weight_path):
        self.model = model_from_json(open(model_path).read())
        print('model loaded from file...')
        self.model.load_weights(model_weight_path)
        print('model weights loaded from file...')

    def load_vocab(self):
        vocab = defaultdict()
        with open(self._vocab_file_path, 'r') as f:
            for line in f.readlines():
                key, value = line.split('\t')
                vocab[key] = value

        return vocab

    def predict(self, test_file, verbose=False):
        try:
            start = time.time()
            self.test = dh.loaddata(test_file, self._word_file_path, self._split_word_file_path, self._emoji_file_path,
                                    normalize_text=True,
                                    split_hashtag=False,
                                    ignore_profiles=True)
            end = time.time()
            if (verbose == True):
                print('test resource loading time::', (end - start))

            self._vocab = self.load_vocab()

            start = time.time()
            tX, tY, tD, tC, tA = dh.vectorize_word_dimension(self.test, self._vocab)
            tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)
            end = time.time()
            if (verbose == True):
                print('test resource preparation time::', (end - start))

            self.__predict_model(tX, self.test)
        except Exception as e:
            print('Error:', e)

    def __predict_model(self, tX, test):
        # calculates output and writes to a file.
        prediction_probability = self.model.predict_proba(tX, batch_size=1, verbose=1)

        y = []
        y_pred = []

        try:
            fd = open(self._output_file + '.analysis', 'w')
            for i, (label) in enumerate(prediction_probability):
                gold_label = test[i][0]
                words = test[i][1]
                dimensions = test[i][2]
                context = test[i][3]
                author = test[i][4]

                predicted = numpy.argmax(prediction_probability[i])

                y.append(int(gold_label))
                y_pred.append(predicted)

                fd.write(str(label[0]) + '\t' + str(label[1]) + '\t'
                         + str(gold_label) + '\t'
                         + str(predicted) + '\t'
                         + ' '.join(words) + '\t'
                         + ' '.join(context))
                fd.write('\n')

            print('precision::', metrics.precision_score(y, y_pred, average='weighted'))
            print('recall::', metrics.recall_score(y, y_pred, average='weighted'))
            print('f_score::', metrics.f1_score(y, y_pred, average='weighted'))

            fd.close()

        except Exception as e:
            print(e)


if __name__ == "__main__":
    basepath = os.getcwd()[:os.getcwd().rfind('/')]

    # offensive content

    # train_file = basepath + '/resource/train/offensive_train.txt'
    train_file = basepath + '/resource/train/hate_speech_kaggle_train.txt'
    validation_file = basepath + '/resource/test/onlineHarassmentDataset.txt'
    test_file = basepath + '/resource/dev/train_english.txt.train'
    word_file_path = basepath + '/resource/word_list_freq.txt'

    split_word_path = basepath + '/resource/word_split.txt'
    emoji_file_path = basepath + '/resource/emoji_unicode_names_final.txt'

    output_file = basepath + '/resource/text_model/TestResults.txt'
    model_file = basepath + '/resource/text_model/weights/'
    vocab_file_path = basepath + '/resource/text_model/vocab_list.txt'

    tr = train_model(train_file, validation_file, word_file_path, split_word_path, emoji_file_path, model_file,
                     vocab_file_path, output_file,
                     model_filename='offensive.json')

    t = test_model(model_file, word_file_path, split_word_path, emoji_file_path, vocab_file_path, output_file)
    t.load_trained_model('offensive.json')
    t.predict(validation_file, verbose=True)

    # hate_speech

    # train_file = basepath + '/resource/train/neither_hate_speech_format.train'
    # validation_file = basepath + '/resource/test/train_english.txt.test'
    # test_file = basepath + '/resource/dev/train_english.txt.train'
    # word_file_path = basepath + '/resource/word_list.txt'
    #
    # output_file = basepath + '/resource/text_model/TestResults.txt'
    # model_file = basepath + '/resource/text_model/weights/'
    # vocab_file_path = basepath + '/resource/text_model/vocab_list.txt'
    #
    # tr = train_model(train_file, validation_file, word_file_path, model_file, vocab_file_path, output_file,model_filename='hate_speech.json')

    # emotion

    # train_file = basepath + '/resource/train/emotion_train.txt'
    # validation_file = basepath + '/resource/test/emotion_train.txt'
    # test_file = basepath + '/resource/dev/train_english.txt.train'
    # word_file_path = basepath + '/resource/word_list.txt'
    #
    # output_file = basepath + '/resource/text_model/TestResults.txt'
    # model_file = basepath + '/resource/text_model/weights/'
    # vocab_file_path = basepath + '/resource/text_model/vocab_list.txt'
    #
    # tr = train_model(train_file, train_file, word_file_path, model_file, vocab_file_path, output_file,
    #                  model_filename='emotion.json')

    # t = test_model(word_file_path, model_file, vocab_file_path, output_file)
    # t.load_trained_model()
    # t.predict(test_file)
