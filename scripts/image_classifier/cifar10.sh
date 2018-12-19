python mural/images_classifier.py --dataset CIFAR10 --model CNN --loss CROSSENTROPY --optimizer SGD --rate 0.01 --epochs 50 --learning VALID_STEPS --imageshow 2

python mural/images_classifier.py --dataset CIFAR10 --model CNN --loss CROSSENTROPY  --learning INFER_MULTI --imageshow 2
