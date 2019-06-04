# Reinforced Data Selection for Domain Adaptation

This is the source code for NAACL-HLT 2019 paper "Incorporating Context and External Knowledge for Pronoun Coreference Resolution".

The readers are welcome to star/fork this repository and use it to train your own model, reproduce our experiment, and follow our future work. Please kindly cite our paper:

```
@inproceedings{liu2019reinforced,
  author    = {Miaofeng Liu and
               Yan Song and
               Hongbin Zou and
               Yan Song},
  title     = {Reinforced Data Selection for Domain Adaptation},
  booktitle = {Proceedings of ACL, 2019},
  year      = {2019}
}
```


#### In this document, we briefly overview how to set up and run selection distribution generator (SDG) to dynamically select data for domain adaptation in three natural language processing tasks.

## Set up and run

Download the raw datasets, [SANCL2012](https://sites.google.com/site/sancl2012/home/shared-task) and [multi-domain Amazon product review](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/) for POS tagging / dependency parsing and sentiment analysis respectively.

- ### Sentiment analysis

  ```bash
  cd sentiment-analysis/sentiment-analysis/src
  python train_AC.py
  ```

- ### POS tagging

  ```bash
  cd POS_tagging/src   
  python all_source_all_target_change.py
  ```

- ### Dependency parsing

  ```bash
  cd PARSing/bist-parser/bmstparser/src/
  python parser-9-20-target_weblogs.py            # e.g. for "weblogs" as target domain    
  ```



## Implementation details of experiments

- ### Sentiment analysis

  ####Requirements

  - Python 3
  - Tensorflow > 0.12
  - Numpy

  ####Training

  Print parameters:

  ```bash
  ./train_AC.py --help
  ```

  ```bash
  optional arguments of used CNN (parameters set to default) :
    -h, --help            show this help message and exit
    --embedding_dim EMBEDDING_DIM
                          Dimensionality of character embedding (default: 128)
    --filter_sizes FILTER_SIZES
                          Comma-separated filter sizes (default: '3,4,5')
    --num_filters NUM_FILTERS
                          Number of filters per filter size (default: 128)
    --l2_reg_lambda L2_REG_LAMBDA
                          L2 regularizaion lambda (default: 0.0)
    --dropout_keep_prob DROPOUT_KEEP_PROB
                          Dropout keep probability (default: 0.5)
    --batch_size BATCH_SIZE
                          Batch Size (default: 64)
    --num_epochs NUM_EPOCHS
                          Number of training epochs (default: 100)
    --evaluate_every EVALUATE_EVERY
                          Evaluate model on dev set after this many steps
                          (default: 100)
    --checkpoint_every CHECKPOINT_EVERY
                          Save model after this many steps (default: 100)
    --allow_soft_placement ALLOW_SOFT_PLACEMENT
                          Allow device soft device placement
    --noallow_soft_placement
    --log_device_placement LOG_DEVICE_PLACEMENT
                          Log placement of ops on devices
    --nolog_device_placement
  
  ```

  Train:

  ```bash
  python train_AC.py
  ```

  ####Evaluating

  ```bash
  ./eval.py --eval_train --checkpoint_dir="./runs/18020311733/checkpoints/"
  ```

  Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data. Run `plot_tsne3.py` to visualize the data representations from the feature extractor for sentiment analysis on the DVD domain.

  ####References

  - [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)

  - [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)


- ### POS tagging

  The based tagger is essentially the Bidirectional Long-Short Term Memory tagger in http://arxiv.org/abs/1604.05529.

  ####Requirements

  - python3 
  - [DyNet 2.0](https://github.com/clab/dynet)

  ####Installation

  Download and install dynet in a directory of your choice DYNETDIR: 

  ```
  mkdir $DYNETDIR
  git clone https://github.com/clab/dynet
  ```

  Follow the instructions in the Dynet documentation (use `-DPYTHON`,
  see http://dynet.readthedocs.io/en/latest/python.html). 

  And compile dynet:

  ```
  cmake .. -DEIGEN3_INCLUDE_DIR=$HOME/tools/eigen/ -DPYTHON=`which python`
  ```

  (if you have a GPU, use: [note: non-deterministic behavior]):

  ```
  cmake .. -DEIGEN3_INCLUDE_DIR=$HOME/tools/eigen/ -DPYTHON=`which python` -DBACKEND=cuda
  ```

  After successful installation open python and import dynet, you can
  test if the installation worked with:

  ```
  >>> import dynet
  [dynet] random seed: 2809331847
  [dynet] allocating memory: 512MB
  [dynet] memory allocation done.
  >>> dynet.__version__
  2.0
  ```

  (You may need to set you PYTHONPATH to include Dynet's `build/python`)

  #####DyNet supports python 3

  The old bilstm-aux had a patch to work with python 3. This
  is no longer necessary, as DyNet supports python 3 as of
  https://github.com/clab/dynet/pull/130#issuecomment-259656695

  #### Example command

  Training the based tagger:

  ```
  python src/bilty.py --dynet-mem 1500 --train data/da-ud-train.conllu --dev data/da-ud-test.conllu --iters 10 --pred_layer 1
  ```

  We integrate training and evaluation into `all_source_all_target.py`.

  #### Embeddings

  The Polyglot embeddings [(Al-Rfou et al.,
  2013)](https://sites.google.com/site/rmyeid/projects/polyglot) can be
  downloaded from [here](http://www.let.rug.nl/bplank/bilty/embeds.tar.gz) (0.6GB)

  #### References

  - [Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss](https://arxiv.org/abs/1604.05529)





- ### Dependency parsing

  The technical details behind the based parser are described in the paper [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198). 

  #### Requirements

  - Python 2.7 interpreter
  - [DyNet library](https://github.com/clab/dynet/tree/master/python) 

  #### Training

  The code requires having a `training.conll` and `development.conll` files formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat). For the based parsers, the graph-based parser acheives an accuracy of 93.8 UAS and the transition-based parser an accuracy of 94.7 UAS on the standard Penn Treebank dataset (Standford Dependencies). The transition-based parser requires no part-of-speech tagging and setting all the tags to NN will produce the expected accuracy.

  To train a based parsing model directly with for either parsing architecture type the following at the command line:

  ```python
  python src/parser.py --dynet-seed 123456789 [--dynet-mem XXXX] --outdir [results directory] --train training.conll --dev development.conll --epochs 30 --lstmdims 125 --lstmlayers 2 [--extrn extrn.vectors] --bibi-lstm
  ```

  We adopt the same external embedding used in [Transition-Based Dependency Parsing with Stack Long Short-Term Memory](http://arxiv.org/abs/1505.08075) which can be downloaded from the authors [github repository](https://github.com/clab/lstm-parser/) and [directly here](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view?usp=sharing).

  ####Parse test data

  The command for parsing a `test.conll` file formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat) with a previously trained model is:
  ```
  python parser.py --predict --outdir [results directory] --test test.conll [--extrn extrn.vectors] --model [trained model file] --params [param file generate during training]
  ```

  The parser will store the resulting conll file in the out directory (`--outdir`).

  We integrate training and parsing into `parser.py`.



  ####References

  - [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198)
  - [Transition-Based Dependency Parsing with Stack Long Short-Term Memory](http://arxiv.org/abs/1505.08075) 

 



