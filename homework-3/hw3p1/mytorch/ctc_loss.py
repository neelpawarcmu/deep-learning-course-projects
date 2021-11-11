import numpy as np
from ctc import *


class CTCLoss(object):
    """CTC Loss class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument:
                blank (int, optional) – blank label index. Default 0.
        """
        # -------------------------------------------->
        # Don't Need Modify
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []

        #modified by neel
        self.ctc = CTC()
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # -------------------------------------------->
        # Don't Need Modify
        return self.forward(logits, target, input_lengths, target_lengths)
        # <---------------------------------------------

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward.

        Computes the CTC Loss.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        loss: scalar
            (avg) divergence between the posterior probability γ(t,r) and the input symbols (y_t^r)

        """
        print('logits', logits.shape)
        print('target', target.shape)
        print('input_lengths', input_lengths)
        print('target_lengths', target_lengths)

        # -------------------------------------------->
        # Don't Need Modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        # <---------------------------------------------

        # -------------------------------------------->
        # Don't Need Modify
        # B = batch_size
        B, _ = target.shape 
        totalLoss = np.zeros(B)
        # <---------------------------------------------

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------
            
            # -------------------------------------------->
            # Truncate the target to target length and logits to input length
            target_truncated_slice = target[b, :target_lengths[b]]
            logits_truncated_slice = logits[:input_lengths[b], b, :]

            # Extend target sequence with blank
            extSymbols, skipConnect = self.ctc.targetWithBlank(target_truncated_slice)

            # Compute forward probabilities and backward probabilities
            alpha = self.ctc.forwardProb(logits_truncated_slice, extSymbols, skipConnect)
            beta = self.ctc.backwardProb(logits_truncated_slice, extSymbols, skipConnect)
            
            # Compute posteriors using total probability function
            gamma = self.ctc.postProb(alpha, beta)
            # print('logits:\n', logits.shape)
            # print('target trunc:\n', target_truncated_slice)
            # print('logits trunc:\n', logits_truncated_slice.round(decimals=2))
            # print('gamma:\n', gamma.round(decimals=2))
            # print(f'b: {b}, gamma shape: {gamma.shape}')
            
            batch_loss = 0
            T, S = gamma.shape
            for t in range(T):
                for s in range(S):
                    loss_at_single_input = - gamma[t,s] * np.log(logits_truncated_slice[t, extSymbols[s]])
                    batch_loss += loss_at_single_input

            # print('batch_loss', batch_loss)
            # print('target_lengths[b]', target_lengths[b])

            totalLoss[b] = batch_loss
            # <---------------------------------------------

        return np.mean(totalLoss)

    def backward(self):
        """CTC loss backard.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        dY: (seqlength, batch_size, len(Symbols))
            derivative of divergence wrt the input symbols at each time.

        """
        # -------------------------------------------->
        # Don't Need Modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)
        print('dY.shape', dY.shape)

        # <---------------------------------------------

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------
            
            
            # -------------------------------------------->
            # Truncate the target to target length and logits to input length
            target_truncated_slice = self.target[b, :self.target_lengths[b]]
            logits_truncated_slice = self.logits[:self.input_lengths[b], b, :]

            # Extend target sequence with blank
            extSymbols, skipConnect = self.ctc.targetWithBlank(target_truncated_slice)

            alpha = self.ctc.forwardProb(logits_truncated_slice, extSymbols, skipConnect)
            beta = self.ctc.backwardProb(logits_truncated_slice, extSymbols, skipConnect)

            gamma = self.ctc.postProb(alpha, beta)

            T, S = gamma.shape
            for t in range(T):
                for s in range(S):
                    dY[t, b, extSymbols[s]] -= gamma[t, s] / logits_truncated_slice[t, extSymbols[s]]
            #
            # <---------------------------------------------
            
        return dY
