Publications:

Yu Cheng, Fei Wang, Ping Zhang, Jianying Hu. Risk Prediction with Electronic Health Records: A Deep Learning Approach. (SDM 2016)

Zhengping Che, Yu Cheng, Zhaonan Sun and Yan Liu. Exploiting Convolutional Neural Network for Risk Prediction with Medical Feature Embedding. (NIPS 2016 ML4HC Workshop)



Directories and Files

Code

/python/deep-cnn/: code for ConvNet training.
Train ConvNet: run_nnet_classification.py
Train baselines: run_baseline_classification.py
Train ConvNet with time interval information: run_nnet_classification_with_time_interval.py


Examples

Some command examples can be found in /scripts.txt (when running on CCC).
Assume code files are in /python/deep-cnn/, record files are in /data/{data}/. Word2vec embedding file is in /data/word2vec/.
The output files will be in the folder /output/{data}/{settings-and-time}/
An example with standard setting
python run_nnet_classification.py --data pre_hf -f 3,4,5 --n_kernels 100 --activation tanh --vocab_embedding_type static --vocab_embedding_size 200
Arguments for calling run_nnet_classification.py:
--data: data folder name.
--filter: size of filter(s). A list of integer splitted by comma.
--n_kernels: # of each filter.
--vocab_embedding_type: either static (embedding is not stable during the ConvNet training), nonstatic (embedding is jointly trained with ConvNet, initialized by the original embedding), or both. (two embeddings are used in the ConvNet, one is static and one is nonstatic.)
--vocab_embedding_size: dimension of embeddings.
--save_features: Add this to save the output from the second last layer of ConvNet. By default it's turned off.
Other arguments (less important) are described in the code file.
Output after calling run_nnet_classification.py:
nnet.dat: cPickled model file.
best_dev_params.{settings-and-so-on}.dat: cPickled best model parameters.
The model can be fully recoverd given the two above files.
cnn_features_{train/valid/test}.npy: output from second last layer.
submission.txt and submission1.txt: Predictions and the ground-truth labels of records in test fold.
/output/summary.txt: accuracy, AUROC, AUPRC, max F1, precision @ top 10/20/30 scores of records in test fold. These statistics also show in stdout.
Arguments for calling run_baseline_classification.py:
--data: same as above.
--feature: The input. Can be bofw (take the bag of words of event indeces), or any combination of sum, min, max (of the embeddings).
--model: LR, SVM, GBT, RF.
--vocab_embedding_size: same as above.