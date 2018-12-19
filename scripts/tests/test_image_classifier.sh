# mnist
python mural/image_classifier.py --dataset MNIST --model CLASSIFIER --loss NLL --optimizer ADAM --rate 0.01 --epochs 2 --learning VALID_STEPS --imageshow 2
python mural/image_classifier.py --dataset MNIST --model MLP --loss CROSSENTROPY --optimizer SGD --rate 0.01 --epochs 2 --learning VALID_STEPS --imageshow 2
python mural/image_classifier.py --dataset MNIST --model MLP --loss CROSSENTROPY --learning INFER_MULTI --imageshow 2

# fashionmnist
python mural/image_classifier.py --dataset FASHIONMNIST --model MLP --loss CROSSENTROPY --optimizer SGD --rate 0.01 --epochs 2 --learning VALID_STEPS --imageshow 2

# cifar10
python mural/image_classifier.py --dataset CIFAR10 --model CNN --loss CROSSENTROPY --optimizer SGD --rate 0.01 --epochs 2 --learning VALID_STEPS --imageshow 2
python mural/image_classifier.py --dataset CIFAR10 --model CNN --loss CROSSENTROPY  --learning INFER_MULTI --imageshow 2

# catsdogs
