python mural/texts_predictor.py --dataset BOOK_ANNA --model CHARRNN --loss CROSSENTROPY --optimizer ADAM --rate 0.001 --epochs 20 --batchsize 128 --seqlength 100 --clip 5 --learning VALID_STEPS --imageloop 1

python mural/texts_predictor.py --dataset BOOK_ANNA --model CHARRNN --learning PREDICT --predict_size 1000 --predict_prime Anna --predict_topk 5
