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

        #####  Attention:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # -------------------------------------------->
        # Don't Need Modify
        # B = batch_size
        B, _ = target.shape 
        totalLoss = np.zeros(B)
        # <---------------------------------------------
        
        #@@@@ Initialize a ctc
        ctc = CTC()

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
            
            # Truncate the target to target length and logits to input length
            target_truncated_slice = target[b, :target_lengths[b]]
            logits_truncated_slice = logits[:input_lengths[b], b, :]

            # Extend target sequence with blank
            target_truncated_slice, skipConnect = ctc.targetWithBlank(target_truncated_slice)

            # Compute forward probabilities and backward probabilities
            alpha = ctc.forwardProb(logits_truncated_slice, target_truncated_slice, skipConnect)
            beta = ctc.backwardProb(logits_truncated_slice, target_truncated_slice, skipConnect)
            
            # Compute posteriors using total probability function
            gamma = ctc.postProb(alpha, beta)
            print('inp:', logits.shape)
            print('inp trunc:', logits_truncated_slice.shape)
            print('gamma:', gamma.shape)

            # -------------------------------------------->
            # Your Code goes here

            raise NotImplementedError
            # <---------------------------------------------

        return totalLoss

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

            # Your Code goes here
            raise NotImplementedError
            # <---------------------------------------------

        return dY
