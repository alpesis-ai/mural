##############################################################################
Mural
##############################################################################

Deep learning all in one using PyTorch Framework.

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

    $ python scripts/data/mnist.py
    $ python scripts/data/fashion_mnist.py

Model Running

::

    # datasets: MNIST: FASHIONMNIST
    # model: CLASSIFIER, CLASSIFIER_DROPOUT
    # optimizer: ADAM, SGD
    # learning: VALID_SINGLE, VALID_STEPS, INFER_SINGLE, INFER_MULTI
    $ python mural/main.py --dataset MNIST --model CLASSIFIER --optimizer ADAM --epochs 50 --learning VALID_STEPS

==============================================================================
Datasets
==============================================================================

- `mnist`_:
- `fashion_minst`_:

.. _`mnist`: http://yann.lecun.com/exdb/mnist/
.. _`fashion_mnist`: https://github.com/zalandoresearch/fashion-mnist
