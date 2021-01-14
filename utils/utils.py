import torch
import matplotlib.pyplot as plt

def show_interpretation(sample, data, target, shape = (1,28,28)):
  channels = shape[0]
  for i in range(len(sample)):
    print(f"Wanted target category : {target[i]}")
    sample_reshaped = sample[i].reshape(shape)
    for k in range(channels):
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(data[i][k], cmap='gray', interpolation='none')
        plt.subplot(1,2,2)
        plt.imshow(sample_reshaped[k], cmap='gray', interpolation='none', vmin=0, vmax=1)
        plt.show()

def save_interpretation(path, sample, data, target, shape = (1,28,28),suffix = ""):
  channels = shape[0]
  for i in range(len(sample)):
    print(f"Wanted target category : {target[i]}")
    sample_reshaped = sample[i].reshape(shape)
    for k in range(channels):
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(data[i][k], cmap='gray', interpolation='none')
        plt.subplot(1,2,2)
        plt.imshow(sample_reshaped[k], cmap='gray', interpolation='none', vmin=0, vmax=1)
        plt.savefig(os.path.join(path,f"{target[i].item()}_{suffix}.jpg"))


def fill_dic(total_dic, dic):
    if len(total_dic.keys())==0:
        for key in dic.keys():
            if isinstance(dic[key], Iterable):
                total_dic[key]=dic[key]
            else :
                total_dic[key] = [dic[key]]

    else :
        for key in dic.keys():
            # print(dic[key])
            if isinstance(dic[key], Iterable):
                total_dic[key].extend(dic[key])
            else :
                total_dic[key].append(dic[key])


    return total_dic

def save_dic(path, dic):
    if not os.path.exists(path):
        os.makedirs(path)

    for key in dic.keys():
        table = dic[key]
        # print(key)
        # print(table)
        # print(np.linspace(0,len(table)-1,len(table)))
        plt.figure(0)
        plt.plot(np.linspace(0,len(table)-1,len(table)),table)
        plt.savefig(os.path.join(path,str(key)+".jpg"))
        plt.clf()