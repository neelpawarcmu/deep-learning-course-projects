import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):  
    def __init__(self, X, Y, offset=1, context=1):
        #load data
        self.X = X
        self.Y = Y
        #create index map for X
        index_map_X = []
        for i, x in enumerate(X):
            for j, xx in enumerate(x):
                index_map_X.append((i,j))
        #create index map for Y
        index_map_Y = []
        for i, y in enumerate(Y):
            for j, yy in enumerate(y):
                index_map_Y.append((i,j))
        #check that number of labels match number of data points
        assert(len(index_map_X) == len(index_map_Y))
        ### Assign data index mapping to self (1 line)
        self.index_map = index_map_X
        #check length
        self.length = len(self.index_map)
        #add context and offset to self
        self.context = context
        self.offset = offset
        #zero pad data as needed for context size = 1 (1-2 lines)
        for i, x in enumerate(self.X):
            self.X[i] = np.pad(x, 
                                ((self.context, self.context), (0, 0)), #ax=0 (row padding), axis=1 (column padding)
                                        'constant', 
                                        constant_values=0)
        '''
        100 elem vector with every elem = melspects 200 * 40 => 210 * 40
        '''

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        ### Get index pair from index map (1-2 lines)
        i, j = self.index_map[index]
        ### Calculate starting timestep using offset and context (1 line)
        start_j = j + self.offset - self.context
        ## Calculate ending timestep using offset and context (1 line)
        end_j = j + self.offset + self.context + 1
        ### Get data at index pair with context (1 line)
        xx = self.X[i][start_j:end_j, :]
        ### Get label at index pair (1 line)
        yy = self.Y[i][j]
        ### Return data at index pair with context and label at index pair (1 line)
        return xx, yy
        
    def collate_fn(self, batch):
        # collate fn not needed
        ### Select all data from batch (1 line)
        batch_x = [x for x,y in batch]
        ### Select all labels from batch (1 line)
        batch_y = [y for x,y in batch]
        ### Convert batched data and labels to tensors (2 lines)
        batch_x = torch.as_tensor(batch_x)
        batch_y = torch.as_tensor(batch_y)
        ### Return batched data and labels (1 line)
        return batch_x, batch_y


#test dataloader using main()
def main():
    X, Y = loadMatrices(tinyDataFlag=True)
    train_dataset = Dataset(X, Y)
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size = 8,
                                            shuffle = False,
                                            collate_fn = train_dataset.collate_fn)
    for i, batch in enumerate(dataloader):
        print("Batch", i, ":\n", batch, "\n")
    

def loadMatrices(tinyDataFlag = True):
    if tinyDataFlag:
        X = np.array([ np.array([[ 1,  2,  3],
                            [ 4,  5,  6],
                            [ 7,  8, 9],
                            [ 10, 11, 12],
                            [ 13,  14,  15],
                            [ 16,  17,  18],
                            [ 19,  20, 21],
                            [ 22, 23, 24]]),
                    np.array([[-1, -2, -3],
                            [-6, -5, -4],
                            [-7, -8, -9],
                            [-10, -11, -12]]) ], dtype=object)

        Y = np.array([ np.array([1, 2, 3, 4, 5, 6, 7, 8]), 
                    np.array([9, 10, 11, 12])], dtype=object)
    else:
        X = np.load('hw1p2/hw1p2-toy-problem/toy_train_data.npy', allow_pickle=True)
        Y = np.load('hw1p2/hw1p2-toy-problem/toy_train_data.npy', allow_pickle=True)
        
    return X, Y


if __name__ == "__main__":
    main()
