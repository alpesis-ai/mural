from texts import tokenize, get_batches

def test_get_batches():
    text = """
           Happy families are all alike; every unhappy family is unhappy in its own way.
           Happy families are all alike; every unhappy family is unhappy in its own way.
           Happy families are all alike; every unhappy family is unhappy in its own way.
           Happy families are all alike; every unhappy family is unhappy in its own way.
           Happy families are all alike; every unhappy family is unhappy in its own way.
           """
    encoded = tokenize(text)

    batch_size = 8
    seq_length = 50
    batches = get_batches(encoded, batch_size, seq_length)
    x, y = next(batches)
    print('x\n', x[:10, :10])
    print('\ny\n', y[:10, :10])


if __name__ =='__main__':
    test_get_batches()
