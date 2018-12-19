python mural/text_predictor.py --dataset BOOK_DUMMY --model CHARRNN --loss CROSSENTROPY --optimizer ADAM --rate 0.001 --epochs 2 --batchsize 18 --seqlength 10 --clip 5 --learning VALID_STEPS --imageloop 1

python mural/text_predictor.py --dataset BOOK_DUMMY --model CHARRNN --learning PREDICT --predict_size 1000 --predict_prime unhappy --predict_topk 5
