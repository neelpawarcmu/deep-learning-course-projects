import numpy as np


def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """
    # Follow the pseudocode from lecture to complete greedy search :-)

    print('SymbolSets\n', SymbolSets)
    print('y_probs\n', y_probs)
    # print('y_probs shape\n', y_probs.shape)

    

    num_symbols_w_blank, Seq_length, batch_size = y_probs.shape
    SymbolSets_w_blank = ['-'] + SymbolSets

    #greedy search
    decoded_batches = [''] * batch_size
    probability = [1] * batch_size
    for s in range(Seq_length):
        for b in range(batch_size):
            slice = y_probs[:,s,b]
            # print(f's:{s}, b: {b}, slice: {slice}')
            idx, prob = np.argmax(slice), np.max(slice)
            probability[b] *= prob
            # print('argmax of slice: ', idx)
            selected_letter = SymbolSets_w_blank[idx]
            print('selected letter:', selected_letter)
            decoded_batches[b] += selected_letter
    print('decoded_sequences', decoded_batches)
    print('probability', probability)
    
    #compress path
    compressed_sequence = [''] * batch_size
    for b in range(batch_size):
        seq = decoded_batches[b]
        for i, char in enumerate(seq):
            if char != '-' and (i==0 or seq[i-1] != seq[i]):
                compressed_sequence[b] += char
    print('compressed_sequence', compressed_sequence)
    # return (forward_path, forward_prob)
    return compressed_sequence[0], probability[0] # return 0th index because we have only batch_size = 1 and expected output is thus a string


##############################################################################


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
    # Follow the pseudocode from lecture to complete beam search :-)
    print('SymbolSets\n', SymbolSets)
    print('y_probs\n', y_probs)
    print('y_probs shape\n', y_probs.shape)

    num_symbols_w_blank, Seq_length, batch_size = y_probs.shape
    for s in range(Seq_length):
        for b in range(batch_size):
            slice = y_probs[:,s,b]
            print(slice)
            top_idxs = slice.argsort()[-BeamWidth:][::-1]
            print(top_idxs)
                

    # return (bestPath, mergedPathScores)
    raise NotImplementedError
