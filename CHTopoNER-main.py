# -*-coding:utf-8 -*-
from tools.layer import *
from tools.utils import add_layer_summary
from tools.train_utils import load_bert_checkpoint
from config import TRAIN_PARAMS
import tensorflow as tf


def reshape_input(input_, params):
    input_ = tf.reshape(input_, [-1, params['max_seq_len'],
                                 int(params['word_enhance_dim'] * params['max_lexicon_len'])])
    return input_


def build_graph(features, labels, params, is_training):
    # print('==========================hello====================')
    """
    roberta + improved softlexicon word enhance + TCNN + CRF
    """
    input_ids = features['token_ids']
    label_ids = features['label_ids']
    input_mask = features['mask']
    segment_ids = features['segment_ids']
    seq_len = features['seq_len']

    embedding = pretrain_bert_embedding(input_ids, input_mask, segment_ids, params['pretrain_dir'],
                                        params['embedding_dropout'], is_training)
    load_bert_checkpoint(params['pretrain_dir'])  # load pretrain bert weight from checkpoint

    # reshape -> batch, max_seq_len ,word_enhance_dim * max_lexicon_len
    softlexicon_ids = reshape_input(features['softlexicon_ids'], params)
    softlexicon_weights = reshape_input(features['softlexicon_weights'], params)

    with tf.variable_scope('word_enhance'):
        # Init word embedding with pretrain word2vec model
        softword_embedding = tf.get_variable(initializer=params['word_embedding'],
                                             dtype=params['dtype'],
                                             name='softlexicon_embedding')
        word_embedding_dim = softword_embedding.shape.as_list()[-1]
        wh_embedding = tf.nn.embedding_lookup(softword_embedding, softlexicon_ids)
        wh_embedding = tf.multiply(wh_embedding, tf.expand_dims(softlexicon_weights, axis=-1))
        wh_embedding = tf.reshape(wh_embedding, [-1, params['max_seq_len'], params['word_enhance_dim'],
                                                 params['max_lexicon_len'], word_embedding_dim])
        wh_embedding = tf.reduce_sum(wh_embedding, axis=3)
        wh_embedding = tf.reshape(wh_embedding, [-1, params['max_seq_len'],
                                                 int(params['word_enhance_dim'] * word_embedding_dim)])
        add_layer_summary('wh_embedding', wh_embedding)

        wh_embedding = tf.layers.dropout(wh_embedding, rate=params['embedding_dropout'],
                                         seed=1234, training=is_training)
        embedding = tf.concat([wh_embedding, embedding], axis=-1)

    lstm_output = bilstm(embedding, params['cell_type'], params['rnn_activation'],
                         params['hidden_units_list'], params['keep_prob_list'],
                         params['cell_size'], seq_len, params['dtype'], is_training)


    lstm_output = tf.layers.dropout(lstm_output, seed=1234, rate=params['embedding_dropout'],
                                      training=is_training)


    idcnn_output = idcnn_layer(embedding, params['num_filters'], params['num_dilations'], params['dilation_rate'], params['activation'], params['drop_out'], is_training)

    con_logits = tf.concat([lstm_output, idcnn_output], axis=-1)

    with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
        logits = tf.layers.dense(con_logits, units=params['label_size'], activation=tf.nn.softmax,
                                      use_bias=True, name='logits')

    add_layer_summary(logits.name, logits)

    trans, log_likelihood = crf_layer(logits, label_ids, seq_len, params['label_size'], is_training)
    pred_ids = crf_decode(logits, trans, seq_len, params['idx2tag'], is_training)
    crf_loss = tf.reduce_mean(-log_likelihood)

    return crf_loss, pred_ids


RNN_PARAMS = {
    'cell_type': 'lstm',
    'cell_size': 1,
    'hidden_units_list': [200], 
    'keep_prob_list': [0.9],
    'rnn_activation': 'tanh',

    # 'filter_list': [10],
    # 'kernel_size_list': [4, 4, 4],
    # 'padding': 'SAME',
    # 'cnn_activation': 'relu',
    # 'cnn_dropout': 0.2,

    'num_filters' : [64, 64, 64, 128],
    'num_dilations' : 4,
    'dilation_rate' : [1, 2, 2, 4],
    'activation' : 'relu',
    'drop_out' : 0.1

}

TRAIN_PARAMS.update(RNN_PARAMS)
TRAIN_PARAMS.update({
    'lr': 5e-6, 
    'diff_lr_times': {'crf': 500,  'logit': 500, 'lstm': 100, 'idcnn':100, 'word_enhance': 100},  
    'embedding_dropout': 0.5,
    'early_stop_ratio': 20 
})
