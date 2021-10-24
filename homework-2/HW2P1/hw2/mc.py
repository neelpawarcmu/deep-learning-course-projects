# Multiple Choice

# Return the answer to the multiple choice question in the handout.
# If you think option c is the correct answer,
# return 'c'

def question_1():
    # [Image] The first hidden layer has 4 filters of kernel-width 2 and stride 2;
    # the second layer has 3 filters of kernel-width 8 and stride 2; the third layer has 2 filters of kernel-width 6 and stride 2
    return 'b'

def question_2():
    # out_width = [(in_width_padded - kernel_dilated) // stride] + 1,
    # where in_width_padded = in_width + 2 * padding, kernel_dilated = (kernel - 1) * (dilation - 1) + kernel
    return 'd'

def question_3():
    # [Image] ip = 100, kernel = 5, stride = 2. Op = ??
    # Example Input: Batch size = 2, In channel = 3, In width = 100
    # Example W: Out channel = 4, In channel = 3, Kernel width = 5
    # Example Out: Batch size = 2, Out channel = 4, Out width = 48
    return 'b'

def question_4():
    #working of numpy.tensordot
    #A = np.arange(30.).reshape(2,3,5)
    #B = np.arange(24.).reshape(3,4,2)
    #C = np.tensordot(A,B, axes = ([0,1],[2,0]))
    return 'a' #random to test

def question_5():
    #lol question
    return 'a'

