python mural/main.py --dataset MNIST --model CLASSIFIER --loss NLL --optimizer ADAM --epochs 50 --learning VALID_STEPS

python mural/main.py --dataset MNIST --model MLP --loss CROSSENTROPY --optimizer SGD --epochs 2 --learning VALID_STEPS
