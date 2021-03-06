##############################################################################
Mural
##############################################################################

Deep learning all in one using PyTorch Framework.

Feature Summary:

- applications:
    - image classifier
    - image generator
    - text predictor
    - text classifier
    - time series predictor
- datasets:
    - image classifier: mnist, fashion mnist, cifar10, catsdogs
    - image generator:
    - text predictor:
    - text classifier:
    - time series predictor:
- CPU/GPU running mode (automatic detection)

==============================================================================
Getting Started
==============================================================================

------------------------------------------------------------------------------
Preparation
------------------------------------------------------------------------------

Prerequisites:

- Python: 3.6.3
- MiniConda

MiniConda

::

    # download miniconda at https://conda.io/miniconda.html
    $ bash ./Miniconda3-latest-MacOSX-x86_64.sh
    $ export PATH=/Users/<username>/miniconda3/bin:$PATH
    
    # conda commands
    $ conda env <create/update> environment.yml
    $ source activate <project_name>
    $ source deactivate

------------------------------------------------------------------------------
Datasets
------------------------------------------------------------------------------

Data Download

::

    # download data into the folder `_data`
    # image classification
    $ python scripts/data/images_mnist.py
    $ python scripts/data/images_fashionmnist.py
    $ python scripts/data/images_cifar10.py

    # image generation
    $ bash scripts/data/images_styletransfer.sh

    # texts
    $ bash scripts/data/texts_books.sh
    $ bash scripts/data/texts_sentiment.sh

Data Notes:

- images:
    - `mnist`_
    - `fashion_mnist`_
    - `cifar`_
    - `cats_and_dogs`_
- texts:
- time series:

.. _`mnist`: http://yann.lecun.com/exdb/mnist/
.. _`fashion_mnist`: https://github.com/zalandoresearch/fashion-mnist
.. _`cifar`: https://www.cs.toronto.edu/~kriz/cifar.html
.. _`cats_and_dogs`: https://www.kaggle.com/c/dogs-vs-cats


Cats & Dogs:

- download data to ``data/cats_dogs``, unzip train.zip to ``data/cats_dogs/train/1``, unzip test1.zip to ``data/cats_dogs/test/1``, to create a new folder inside train and test for adapting to ``torchvision.datasets.ImageFolder()``, otherwise, it could not be loaded.


==============================================================================
Image Classifier
==============================================================================

------------------------------------------------------------------------------
How it runs
------------------------------------------------------------------------------

Parameters

::

    # datasets: MNIST, FASHIONMNIST, CIFAR10, CATSDOGS
    # learning: VALID_SINGLE, VALID_STEPS, INFER_SINGLE, INFER_MULTI
    # model: CLASSIFIER, CLASSIFIER_DROPOUT, DENSENET_TRANS
    # loss: NLL, CROSSENTROPY
    # optimizer: ADAM, SGD
    # rate: e.g. 0.01
    # learning: VALID_SINGLE, VALID_STEPS, INFER_SINGLE, INFER_MULTI
    # imageshow: 0 - not shown, 1 - single, 2 - multi, 3 -detail
    $ python mural/classifier.py -h


Train

::

    $ python mural/classifier.py
      --dataset CIFAR10
      --model CNN
      --loss CROSSENTROPY
      --optimizer SGD
      --rate 0.01
      --epochs 30
      --learning VALID_STEP
      --imageshow 2

Test

::

    $ python mural/classifier.py
      --dataset CIFAR10
      --model CNN
      --loss CROSSENTROPY
      --learning INFER_MULTI

------------------------------------------------------------------------------
Models
------------------------------------------------------------------------------

==============================================================================
Image Generator
==============================================================================

------------------------------------------------------------------------------
How it runs
------------------------------------------------------------------------------

Style Transfer

::

    $ python mural/generator.py
      --model VGG19_FEATURES
      --optimizer ADAM
      --rate 0.03
      --epochs 2000
      --imageloop 400

------------------------------------------------------------------------------
Models
------------------------------------------------------------------------------

- `Image Style Transfer Using Convolutional Neural Networks`_

.. _`Image Style Transfer Using Convolutional Neural Networks`: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

==============================================================================
Text Predictor
==============================================================================

------------------------------------------------------------------------------
How it runs
------------------------------------------------------------------------------

Parameters

::

    # common:
    # - dataset: BOOK_DUMMY, BOOK_ANNA
    # - learning: VALID_STEPS, PREDICT
    # - model: CHARRNN
    
    # train:
    # - loss: CROSSENTROPY
    # - optimizer: ADAM
    # - rate: 
    # - epochs:
    # - batchsize
    # - seqlength
    # - clip
    
    # test:
    # - predict_size:
    # - predict_prime
    # - predict_topk
    
    $ python mural/text_predictor.py -h
     

Train

::

    $ python mural/text_predictor.py
      --dataset BOOK_ANNA
      --model CHARRNN
      --loss CROSSENTROPY
      --optimizer ADAM
      --rate 0.001
      --epochs 20
      --batchsize 128
      --seqlength 100
      --clip 5
      --learning VALID_STEPS
      --imageloop 10


Predict

::

    $ python mural/text_predictor.py
      --dataset BOOK_DUMMY
      --model CHARRNN
      --learning PREDICT
      --predict_size 1000
      --predict_prime Anna
      --predict_topk 5

------------------------------------------------------------------------------
Models
------------------------------------------------------------------------------

- `CharRNN: The Unreasonable Effectiveness of Recurrent Neural Networks`_

.. _`CharRNN: The Unreasonable Effectiveness of Recurrent Neural Networks`: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

==============================================================================
Text Classifier
==============================================================================

------------------------------------------------------------------------------
How it runs
------------------------------------------------------------------------------

------------------------------------------------------------------------------
Models
------------------------------------------------------------------------------

==============================================================================
Time Series Predictor
==============================================================================

------------------------------------------------------------------------------
How it runs
------------------------------------------------------------------------------

::

    $ python mural/timeseries_predictor.py

------------------------------------------------------------------------------
Models
------------------------------------------------------------------------------

