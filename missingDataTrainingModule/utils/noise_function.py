import numpy as np 



class NoiseFunction():
    def __init__(self):
        return True

    def __str__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

class GaussianNoise(NoiseFunction):
    def __init__(self, sigma = 1.0, regularize = False):
        self.sigma = sigma
        self.regularize = regularize

    def __str__(self):
        return f"GaussianNoise_Sigma{self.sigma}_regularize{self.regularize}"

    def __call__(self, img):
        noise = np.random.normal(0, self.sigma, np.shape(img))
        
        img_noised = img + noise
        if self.regularize :
            img_noised =torch.where(img_noised>torch.max(img),
                 2*img-img_noised, img_noised
                )

            img_noised =torch.where(img_noised<torch.min(img),
                 2* img -img_noised, img_noised
                )
        return img_noised


class DropOutNoise(NoiseFunction):
    def __init__(self, pi=0.5):
        self.pi = pi
        
    def __str__(self):
        return f"DropOutNoise_Pi{self.pi}"

    def __call__(self, img, get_mask = False):
        noise = np.random.binomial(1,self.pi, size = np.shape(img))
        noise = torch.tensor(noise)
        img_noised = torch.where(noise>=0.5, torch.zeros(img.shape), img)
        if get_mask :
            return img_noised, noise
        return img_noised
    

