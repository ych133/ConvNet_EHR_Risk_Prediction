# coding=utf-8

import argparse
import cPickle
from datetime import datetime
import os
import time
import warnings
warnings.filterwarnings("ignore")  # TODO remove

import numpy as np
import pandas as pd
from sklearn import metrics
import theano
from theano import tensor as T
from tqdm import tqdm

from evaluation import maxf1, topKPrecision
import nn_layers
import sgd_trainer

### THEANO DEBUG FLAGS
theano.config.optimizer = 'fast_run'
theano.config.exception_verbosity = 'high'


def main():
    ZEROUT_DUMMY_WORD = True

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data', type=unicode,
                            help='E.g.: pre, pre_hf, etc.')
    arg_parser.add_argument('--filter', '-f', type=unicode, default='3,4,5',
                            help='E.g.: 3 or 3,4,5')
    arg_parser.add_argument('--n_kernels', type=int, default=100,
                            help='# of kernels (filters)')
    arg_parser.add_argument('--n_epochs', type=int, default=25)
    arg_parser.add_argument('--batch_size', type=int, default=50)
    arg_parser.add_argument('--dropout_rate', type=float, default=0.5)
    arg_parser.add_argument('--vocab_embedding_type', type=unicode,
                            default='both', help='both/static/nonstatic')
    arg_parser.add_argument('--vocab_embedding_size', type=int,
                            default='50', help='50/200/500/800')
    arg_parser.add_argument('--L2_embedding', type=float, default=0.)
    arg_parser.add_argument('--L2_conv', type=float, default=0.)
    arg_parser.add_argument('--L2_linear', type=float, default=0.)
    arg_parser.add_argument('--activation', type=unicode, default='tanh')
    arg_parser.add_argument('--input_type', type=unicode, default='index',
                            help='E.g.: index (use a lookup layer), embed.')
    arg_parser.set_defaults(save_nn_features=False)
    arg_parser.add_argument('--save_features', dest='save_nn_features', 
                            action='store_true', 
                            help='Save outputs from second last layer')
                            
    args = arg_parser.parse_args()
    data_name = args.data
    filter_widths = [int(a) for a in args.filter.split(',')]
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    dropout_rate = args.dropout_rate
    vocab_embedding_type = args.vocab_embedding_type
    vocab_embedding_size = args.vocab_embedding_size
    L2_embedding = args.L2_embedding
    L2_conv = args.L2_conv
    L2_linear = args.L2_linear
    nkernels = args.n_kernels
    activation_str = args.activation
    input_type = args.input_type    
    save_nn_features = args.save_nn_features
    
    ## Load data
    data_dir = '../../data/{}'.format(data_name)
    embedding_dir = '../../data/word2vec'
    if input_type == 'index':
        x_train = np.load(os.path.join(data_dir, 'train_input.npy'))
        x_dev = np.load(os.path.join(data_dir, 'valid_input.npy'))
        x_test = np.load(os.path.join(data_dir, 'test_input.npy'))
    elif input_type == 'embed':
        x_train = np.load(os.path.join(
            data_dir, 'train_embed_{}.npy'.format(vocab_embedding_size)))
        x_dev = np.load(os.path.join(
            data_dir, 'valid_embed_{}.npy'.format(vocab_embedding_size)))
        x_test = np.load(os.path.join(
            data_dir, 'test_embed_{}.npy'.format(vocab_embedding_size)))
    y_train = np.load(os.path.join(data_dir, 'train_label.npy'))
    y_dev = np.load(os.path.join(data_dir, 'valid_label.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_label.npy'))
    y_candidates = np.unique(np.concatenate((y_train, y_dev, y_test)))
    n_y_class = len(y_candidates)
    # for multi class label, from (0, 1, 3, 7, ..) to (0, 1, 2, 3, ...)    
    y_train = np.array([np.where(y_candidates==yy)[0][0] for yy in y_train], 
                        dtype='int32')
    y_dev = np.array([np.where(y_candidates==yy)[0][0] for yy in y_dev], 
                      dtype='int32')
    y_test = np.array([np.where(y_candidates==yy)[0][0] for yy in y_test], 
                       dtype='int32')
    if n_y_class > 2:
        y_train_foreval = np.zeros([len(y_train), n_y_class])
        y_train_foreval[np.arange(len(y_train)), y_train] = 1
        y_dev_foreval = np.zeros([len(y_dev), n_y_class])
        y_dev_foreval[np.arange(len(y_dev)), y_dev] = 1
        y_test_foreval = np.zeros([len(y_test), n_y_class])
        y_test_foreval[np.arange(len(y_test)), y_test] = 1
    else:
        y_train_foreval = np.array(y_train > 0, dtype=int)
        y_dev_foreval = np.array(y_dev > 0, dtype=int)
        y_test_foreval = np.array(y_test > 0, dtype=int)
    
    print 'y_train', np.unique(y_train, return_counts=True),
    print 'y_dev', np.unique(y_dev, return_counts=True)
    print 'y_test', np.unique(y_test, return_counts=True)
    print 'x_train', x_train.shape, x_train.dtype, theano.config.floatX
    print 'x_dev', x_dev.shape
    print 'x_test', x_test.shape

    np_rng = np.random.RandomState()
    x_max_sent_size = x_train.shape[1]
    if input_type == 'index':
        ## Load word2vec embeddings
        fname = os.path.join(embedding_dir, 
                             'word2vec_{}.npy'.format(vocab_embedding_size))
        print "Loading word embeddings from", fname
        vocab_emb = np.asarray(np.load(fname), dtype=theano.config.floatX)
        ndim = vocab_emb.shape[1]
        dummy_word_idx = vocab_emb.shape[0] - 1
        print "Word embedding matrix size:", vocab_emb.shape, type(vocab_emb), vocab_emb.dtype
        print 'dummy word index:', dummy_word_idx
    elif input_type == 'embed':
        ndim = x_train.shape[2]
        assert ndim == vocab_embedding_size, \
            'n_dim {} should be the same as emb_size {}'.format(ndim, vocab_embedding_size)

    if input_type == 'index':
        x = T.lmatrix('x')
    else:
        x = T.dtensor3('x_emb')
    y = T.ivector('y')

    ## Settings
    n_out = n_y_class if n_y_class > 2 else 1
    max_norm = 0
    print 'batch_size', batch_size
    print 'n_epochs', n_epochs
    print 'dropout_rate', dropout_rate
    print 'max_norm', max_norm
    print 'n_out', n_out
    print 'filter_widths', filter_widths
    
    reg_str = 'L2emb{}L2conv{}L2linear{}'.format(args.L2_embedding,
                                                 args.L2_conv, args.L2_linear)
    
    setting_str = 'filter={filter};n_f={n_f};activation={activation};' \
                  'emb_size={emb_size};emb_type={emb_type};reg={reg};' \
                  ''.format(filter=args.filter, n_f=args.n_kernels,
                            activation=args.activation, 
                            emb_size=args.vocab_embedding_size, 
                            emb_type=args.vocab_embedding_type,
                            reg=reg_str)
    ts_str = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
    nnet_outdir_pattern = ('../../output/{data}/{setting};time={time}')
    nnet_outdir = nnet_outdir_pattern.format(data=data_name, 
                                             setting=setting_str, time=ts_str)
    if not os.path.exists(nnet_outdir):
        os.makedirs(nnet_outdir)

    ## Conv layer.
    activation = T.tanh if activation_str == 'tanh' else T.nnet.relu
    k_max = 1
    num_input_channels = 1
    # not all of the following 3 layers are used: 
    if input_type == 'index':
        lookup_table_static = nn_layers.LookupTableFastStatic(
            W=vocab_emb, pad=max(filter_widths)-1)
        lookup_table_nonstatic = nn_layers.LookupTableFast(
            W=vocab_emb, pad=max(filter_widths)-1, borrow=False)
    elif input_type == 'embed':
        lookup_table_static = nn_layers.EmbeddingInput(
        pad=max(filter_widths)-1)
    # This is the input shape to the conv layer, not the first layer.
    input_shape = (batch_size, num_input_channels,
                   x_max_sent_size + 2*(max(filter_widths)-1), ndim)
    tconv_layers = []
    for i_width, filter_width in enumerate(filter_widths):
        filter_shape = (nkernels, num_input_channels, filter_width, ndim)
        conv = nn_layers.Conv2dLayer(
            rng=np_rng, filter_shape=filter_shape, input_shape=input_shape)
        non_linearity = nn_layers.NonLinearityLayer(
            b_size=filter_shape[0], activation=activation)
        conv_static = nn_layers.FeedForwardNet(layers=[conv, non_linearity])
        if vocab_embedding_type == 'nonstatic':
            conv_nonstatic = nn_layers.FeedForwardNet(layers=[conv, 
                                                              non_linearity])
        else:
            conv_nonstatic = nn_layers.CopiedLayer(conv_static)
        if i_width == 0:
            tc_static = nn_layers.FeedForwardNet(
                layers=[lookup_table_static, conv_static])
            if input_type  == 'index':
                tc_nonstatic =  nn_layers.FeedForwardNet(
                    layers=[lookup_table_nonstatic, conv_nonstatic])
        else:
            tc_static = nn_layers.FeedForwardNet(
                layers=[nn_layers.CopiedLayer(lookup_table_static), 
                        conv_static])
            if input_type  == 'index':
                tc_nonstatic =  nn_layers.FeedForwardNet(
                    layers=[nn_layers.CopiedLayer(lookup_table_nonstatic), 
                            conv_nonstatic])
        if vocab_embedding_type == 'both':
            tc_multichannel = nn_layers.SumMergeLayer(
                layers=[tc_static, tc_nonstatic])
        elif vocab_embedding_type == 'static':
            tc_multichannel = tc_static
        elif vocab_embedding_type == 'nonstatic':
            tc_multichannel = tc_nonstatic
        pooling = nn_layers.KMaxPoolLayer(k_max=k_max)
        tconv2dNonLinearMaxPool = nn_layers.FeedForwardNet(
            layers=[tc_multichannel, pooling])
        tconv_layers.append(tconv2dNonLinearMaxPool)

    join_layer = nn_layers.ParallelLayer(layers=tconv_layers)
    flatten_layer = nn_layers.FlattenLayer()
    nnet = nn_layers.FeedForwardNet(
        layers=[join_layer,
                flatten_layer,
                ])
    nnet.set_input(x)

    logistic_n_in = nkernels * len(filter_widths) * k_max
    dropout = nn_layers.DropoutLayer(rng=np_rng, p=dropout_rate)
    dropout.set_input(nnet.output)
    if n_out > 2:
        classifier = nn_layers.LogisticRegression(n_in=logistic_n_in, 
                                                  n_out=n_out)
    else:
        classifier = nn_layers.BinaryLogisticRegression(n_in=logistic_n_in)
    classifier.set_input(dropout.output)

    train_nnet = nn_layers.FeedForwardNet(
        layers=[nnet, dropout, classifier],
        name="Training nnet")
    test_nnet = train_nnet
    print 'train_nnet:\n{}'.format(train_nnet)

    params = train_nnet.params

    nnet_fname = os.path.join(nnet_outdir, 'nnet.dat')
    print "Saving to", nnet_fname
    cPickle.dump([train_nnet, test_nnet], 
                 open(nnet_fname, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    with open(os.path.join(nnet_outdir, 'model_str.txt'), 'w') as f:
        f.write(str(train_nnet))
    total_params = sum([np.prod(param.shape.eval()) for param in params])
    print 'Total params number:', total_params

    cost = train_nnet.layers[-1].training_cost(y)
    predictions = test_nnet.layers[-1].y_pred
    predictions_prob = test_nnet.layers[-1].p_y_given_x[:]
    second_last_features = test_nnet.layers[-3].output
    
    ## Add L_2 regularization
    print "Regularizing nnet weights: ",
    for w in train_nnet.weights:
        if w.name.startswith('W_emb'):
            L2_reg_w = L2_embedding
        elif w.name.startswith('W_conv1d'):
            L2_reg_w = L2_conv
        elif w.name.startswith('W_softmax'):
            L2_reg_w = L2_linear
        elif w.name == 'W':
            L2_reg_w = 0.
        print '{}:{}, '.format(w.name, L2_reg_w),
        cost += T.sum(w**2) * L2_reg_w
    print ''

    if input_type == 'index':
        batch_x = T.lmatrix('batch_x')
    elif input_type == 'embed':
        batch_x = T.dtensor3('batch_x_emb')
    batch_y = T.ivector('batch_y')

    updates = sgd_trainer.get_adadelta_updates(cost, params, 
                                               rho=0.95, eps=1e-6, 
                                               max_norm=max_norm, 
                                               word_vec_name='W_emb')
    inputs_pred = [batch_x,]
    givens_pred = {x: batch_x,}
    inputs_train = [batch_x,
                    batch_y,]
    givens_train = {x: batch_x,
                    y: batch_y,}

    train_fn = theano.function(inputs=inputs_train,
                               outputs=cost,
                               updates=updates,
                               givens=givens_train)
    pred_fn = theano.function(inputs=inputs_pred,
                              outputs=predictions,
                              givens=givens_pred)
    pred_prob_fn = theano.function(inputs=inputs_pred,
                                   outputs=predictions_prob,
                                   givens=givens_pred)
    features_fn = theano.function(inputs=inputs_pred,
                                  outputs=second_last_features,
                                  givens=givens_pred)

    def predict_batch(batch_iterator):
        preds = np.concatenate(
            [pred_fn(batch_data[0]) for batch_data in batch_iterator])
        return preds[:batch_iterator.n_samples]

    def predict_prob_batch(batch_iterator):
        preds = np.concatenate(
            [pred_prob_fn(batch_data[0]) for batch_data in batch_iterator])
        return preds[:batch_iterator.n_samples]

    def get_features_batch(batch_iterator):
        features = np.concatenate(
            [features_fn(batch_data[0]) for batch_data in batch_iterator])
        return features[:batch_iterator.n_samples]

    train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        np_rng, [x_train, y_train], batch_size=batch_size, randomize=True)
    train_set_iterator_eval = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        np_rng, [x_train, y_train], batch_size=batch_size, randomize=False)
    dev_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        np_rng, [x_dev, y_dev], batch_size=batch_size, randomize=False)
    test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(
        np_rng, [x_test, y_test], batch_size=batch_size, randomize=False)

    print "Zero out dummy word:", ZEROUT_DUMMY_WORD
    if ZEROUT_DUMMY_WORD:
        W_emb_list = [w for w in params if w.name == 'W_emb']
        zerout_dummy_word = theano.function(
            [], 
            updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list]
            )

    best_dev_auc = -np.inf
    epoch = 0
    timer_train = time.time()
    no_best_dev_update = 0
    num_train_batches = len(train_set_iterator)
    best_params = [np.copy(p.get_value(borrow=True)) for p in params]
    for i, p in enumerate(best_params):
        print i, p.shape,
        print best_params[i].sum()
    while epoch < n_epochs:
        timer = time.time()
        for i, (x, y) in enumerate(tqdm(train_set_iterator), 1):
            train_fn(x, y)
    
            # Make sure the null word embedding always remains zero
            if ZEROUT_DUMMY_WORD:
                zerout_dummy_word()
        
            if i % 10 == 0 or i == num_train_batches:
              y_pred_dev = predict_prob_batch(dev_set_iterator)
              print y_dev_foreval.shape, y_pred_dev.shape
              dev_auc = metrics.roc_auc_score(y_dev_foreval, y_pred_dev) * 100
              if dev_auc > best_dev_auc:
                y_pred = predict_prob_batch(test_set_iterator)
                test_auc = metrics.roc_auc_score(y_test_foreval, y_pred) * 100
                print ('epoch: {} batch: {} dev auc: {:.4f}; '
                       'best_dev_auc: {:.4f}; test_auc: {:.4f}'
                       .format(epoch, i, dev_auc, best_dev_auc, test_auc))
                best_dev_auc = dev_auc
                best_params_pre = best_params
                best_params = [
                    np.copy(p.get_value(borrow=True)) for p in params]
                no_best_dev_update = 0
                for i, p in enumerate(best_params):
                    print i,p.shape,'\t\t\t',
                    print np.array_equal(best_params[i],best_params_pre[i]),
                    print '\t\t\t',
                    print best_params[i].sum()
                print
        if no_best_dev_update >= 3:
            print "Quitting after of no update of the best score on dev set",
            print no_best_dev_update
            break
        print ('epoch {} took {:.4f} seconds'
               .format(epoch, time.time() - timer))
        epoch += 1
        no_best_dev_update += 1

    print('Training took: {:.4f} seconds'.format(time.time() - timer_train))
    for i, param in enumerate(best_params):
        params[i].set_value(param, borrow=True)


    y_pred_train = predict_batch(train_set_iterator_eval)
    y_pred_prob_train = predict_prob_batch(train_set_iterator_eval)
    y_pred_dev = predict_batch(dev_set_iterator)
    y_pred_prob_dev = predict_prob_batch(dev_set_iterator)
    y_pred_test = predict_batch(test_set_iterator)
    y_pred_prob_test = predict_prob_batch(test_set_iterator)
    
    print 'Train:'
    print 'acc is:', metrics.accuracy_score(y_train, y_pred_train)
    print 'auc is:', metrics.roc_auc_score(y_train_foreval, y_pred_prob_train)
    print 'prc is:', metrics.average_precision_score(y_train_foreval, y_pred_prob_train)
    print 'maxf1 is:', maxf1(y_train_foreval, y_pred_prob_train)
    print 'prec @ 10/20/30:', topKPrecision(y_train_foreval, y_pred_prob_train)
    
    print 'Valid:'
    print 'acc is:', metrics.accuracy_score(y_dev, y_pred_dev),
    print 'auc is:', metrics.roc_auc_score(y_dev_foreval, y_pred_prob_dev)
    print 'prc is:', metrics.average_precision_score(y_dev_foreval, y_pred_prob_dev)
    print 'maxf1 is:', maxf1(y_dev_foreval, y_pred_prob_dev)
    print 'prec @ 10/20/30:', topKPrecision(y_dev_foreval, y_pred_prob_dev)
    
    print 'Test:'
    test_acc = metrics.accuracy_score(y_test, y_pred_test)
    test_auc = metrics.roc_auc_score(y_test_foreval, y_pred_prob_test)
    test_prc = metrics.average_precision_score(y_test_foreval, y_pred_prob_test)
    test_maxf1 = maxf1(y_test_foreval, y_pred_prob_test)
    test_prec = topKPrecision(y_test_foreval, y_pred_prob_test)
    
    print 'acc is:', test_acc
    print 'auc is:', test_auc
    print 'prc is:', test_prc
    print 'maxf1 is:', test_maxf1
    print 'prec @ 10/20/30:', test_prec
    
    with open('../../output/summary.txt', 'a') as f:
        f.write(data_name + '\t' + setting_str + '\t')
        f.write('\t'.join([str(a) for a in \
            [test_acc, test_auc, test_prc, test_maxf1, test_prec]]))
        f.write('\n')
    
    fname = os.path.join(nnet_outdir,
                         ('best_dev_params.epoch={:02d};batch={:05d};'
                          'dev_auc={:.2f}.dat'
                          .format(epoch, i, best_dev_auc)))
    cPickle.dump(best_params, open(fname, 'wb'), 
                 protocol=cPickle.HIGHEST_PROTOCOL)
    pred_txt_name_suffix = ('.epoch={:02d};batch={:05d};'
                            'dev_auc={:.2f}.predictions.txt'
                            .format(epoch, i, best_dev_auc))
    np.savetxt(os.path.join(nnet_outdir, 'train' + pred_txt_name_suffix),
               y_pred_prob_train)
    np.savetxt(os.path.join(nnet_outdir, 'valid' + pred_txt_name_suffix),
               y_pred_prob_dev)
    np.savetxt(os.path.join(nnet_outdir, 'test' + pred_txt_name_suffix),
               y_pred_prob_test)
               
    if save_nn_features:
        y_features_train = get_features_batch(train_set_iterator_eval)
        y_features_dev = get_features_batch(dev_set_iterator)
        y_features_test = get_features_batch(test_set_iterator)
        np.save(os.path.join(nnet_outdir, 'cnn_features_train.npy'), y_features_train)
        np.save(os.path.join(nnet_outdir, 'cnn_features_dev.npy'), y_features_dev)
        np.save(os.path.join(nnet_outdir, 'cnn_features_test.npy'), y_features_test)

    N = len(y_pred_test)
    df_submission = pd.DataFrame(
        index=np.arange(N), 
        columns=['docno', 'label','pred'] + \
                ['p' + str(i+1) for i in xrange(n_out)])
    df_submission['docno'] = np.arange(N)
    df_submission['label'] = y_test
    df_submission['pred'] = y_pred_test
    if n_out > 1:
        for i in xrange(n_out):
            df_submission['p' + str(i+1)] = y_pred_prob_test[:, i]
    else:
        df_submission['p1'] = y_pred_prob_test

    df_submission.to_csv(os.path.join(nnet_outdir, 'submission.txt'), 
                         header=True, index=True, sep=' ')
    df_submission.to_csv(os.path.join(nnet_outdir, 'submission1.txt'), 
                         header=False, index=False, sep=' ')
    print nnet_outdir
    print vocab_emb.shape

    print 'epoch', epoch


if __name__ == '__main__':
    main()