# Reinforced Training Data Selection for Domain Adaptation

This is the implementation code for paper "Reinforced Training Data Selection for Domain Adaptation".

The readers are welcome to star/fork this repository and use it to train your own model, modify our experiments. Please kindly cite our paper:

```
@inproceedings{liu2019reinforced,
  author    = {Lewis Liu and
               Yan Song and
               Hongbin Zou and
               Tong Zhang},
  title     = {Reinforced Training Data Selection for Domain Adaptation},
  year      = {2019}
}
```


#### In this document, we briefly illustrate how to set up and run selection distribution generator (SDG) to dynamically select data for multiple source domain adaptation in natural language processing tasks.

## Set up and run

Download the raw datasets from [SANCL2012](https://sites.google.com/site/sancl2012/home/shared-task).

- ### Sentiment analysis

  ```bash
  cd sentiment-analysis/sentiment-analysis/src
  python train_AC.py
  ```



## Implementation details of experiments

- ### Sentiment analysis

  #### Requirements

  - Python 3
  - Tensorflow > 0.12
  - Numpy

  #### Training

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

  #### Evaluating

  ```bash
  ./eval.py --eval_train --checkpoint_dir="./runs/18020311733/checkpoints/"
  ```

  Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data. Run `plot_tsne3.py` to visualize the data representations from the feature extractor for sentiment analysis on the DVD domain.

  #### References

  - [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)

  - [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)




 



