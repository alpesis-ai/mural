##############################################################################
Mural
##############################################################################

Deep learning all in one using PyTorch Framework.

Feature Summary:

- data types:
    - images: image classifier, image generator
    - texts: text classifier, text generator
    - time series: timeseries predictor
- CPU/GPU running mode (automatic detection)

==============================================================================
How it runs
==============================================================================

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
    $ bash scripts/data/texts_charrcnn.sh
    $ bash scripts/data/texts_sentiment.sh

------------------------------------------------------------------------------
Image Classifier
------------------------------------------------------------------------------

Model Parameters

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


Model Train

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

Model Test

::

    $ python mural/classifier.py
      --dataset CIFAR10
      --model CNN
      --loss CROSSENTROPY
      --learning INFER_MULTI

------------------------------------------------------------------------------
Image Generator
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
Text Classifier
------------------------------------------------------------------------------


------------------------------------------------------------------------------
Text Generator
------------------------------------------------------------------------------

------------------------------------------------------------------------------
Time Series
------------------------------------------------------------------------------


==============================================================================
Datasets
==============================================================================

------------------------------------------------------------------------------
Image Classifer
------------------------------------------------------------------------------

- `mnist`_
- `fashion_mnist`_
- `cifar`_
- `cats_and_dogs`_

.. _`mnist`: http://yann.lecun.com/exdb/mnist/
.. _`fashion_mnist`: https://github.com/zalandoresearch/fashion-mnist
.. _`cifar`: https://www.cs.toronto.edu/~kriz/cifar.html
.. _`cats_and_dogs`: https://www.kaggle.com/c/dogs-vs-cats


Cats & Dogs:

- download data to ``data/cats_dogs``, unzip train.zip to ``data/cats_dogs/train/1``, unzip test1.zip to ``data/cats_dogs/test/1``, to create a new folder inside train and test for adapting to ``torchvision.datasets.ImageFolder()``, otherwise, it could not be loaded.


------------------------------------------------------------------------------
Image Generator
------------------------------------------------------------------------------

- `Image Style Transfer Using Convolutional Neural Networks`_

.. _`Image Style Transfer Using Convolutional Neural Networks`: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf


------------------------------------------------------------------------------
Text Classifier
------------------------------------------------------------------------------


------------------------------------------------------------------------------
Text Generator
------------------------------------------------------------------------------

------------------------------------------------------------------------------
Time Series
------------------------------------------------------------------------------
