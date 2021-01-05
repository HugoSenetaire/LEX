import numpy as np

def calculate_pi_dimension(input_size, stride):
    if len(input_size)==1:
        input_size = (1, input_size, input_size)
    
    
    nb_patch_x = int(np.ceil(input_size[1]/stride[0]))
    nb_patch_y = int(np.ceil(input_size[2]/stride[1]))

    # nb_patch_total = nb_patch_x * nb_patch_y
    
    return nb_patch_x, nb_patch_y