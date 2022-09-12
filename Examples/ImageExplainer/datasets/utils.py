import numpy as np

def create_panels(left_data, right_data, target_left, target_right, random_panels= False, target = "right",):
        Xpanels = np.zeros((*(np.shape(left_data)[:-1]), 2*np.shape(left_data)[-1]))
        quadrant = np.zeros((np.shape(left_data)[0], 1 , np.shape(left_data)[2], 2*np.shape(left_data)[-1]))

        if random_panels :
            i = np.random.binomial(1, 0.5, size = len(left_data),)  # i is not nevi here and target is nevi
        else :
            i = np.zeros(left_data.shape[0], dtype = np.int64)
        j = 1 - i
        # size_dataset = np.arange(len(i), dtype=np.int64)

        # Xpanels[size_dataset, :, :, i*28:(i+1)*28,] = left_data
        # Xpanels[size_dataset, :, :, j*28:(j+1)*28,] = right_data
        # if target == "left" : #Target is left
        #     quadrant[size_dataset, :, :, i*28:(i+1)*28,] = 1 
        #     ypanels = target_left
        # if target == "right" : #Target is left
        #     quadrant[size_dataset, :, :, j*28:(j+1)*28,] = 1 
        #     ypanels = target_right
        for k in range(len(i)):
            Xpanels[k, :, :, i[k]*28:(i[k]+1)*28,] = left_data[k]
            Xpanels[k, :, :, j[k]*28:(j[k]+1)*28,] = right_data[k]
            if target == "left" : #Target is left
                quadrant[k, :, :, i[k]*28:(i[k]+1)*28,] = 1 
            else :
                quadrant[k, :, :, j[k]*28:(j[k]+1)*28,] = 1 

        if target == "left" :
            ypanels = target_left
        else :
            ypanels = target_right
        return Xpanels, ypanels, quadrant


def create_validation(data, target, true_selection = None, size = 0.8):
    size_train = int(size*len(data))
    index_train = np.random.choice(np.arange(len(data)), size_train, replace = False)
    index_val = np.setdiff1d(np.arange(len(data)), index_train)

    data_train = data[index_train]
    target_train = target[index_train]
    data_val = data[index_val]
    target_val = target[index_val]

    if true_selection is not None :
        true_selection_train = true_selection[index_train]
        true_selection_val = true_selection[index_val]
    else :
        true_selection_train = None
        true_selection_val = None

    return data_train, target_train, true_selection_train, data_val, target_val, true_selection_val
    