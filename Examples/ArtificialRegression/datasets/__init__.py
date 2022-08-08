from .loader_artificial import LoaderArtificial
from .circle_dataset import CircleDataset
from .gaussian_dataset import SwitchFeature, create_toeplitz_covariance, create_blockwise_covariance, ExpProdGaussianDataset, ExpSquaredSumGaussianDataset, DiagGaussianDataset, ExpSquaredSumGaussianDatasetV2
from .realx_dataset import S_1, S_2, S_3
from .realx_dataset_same_codeL2x import Syn1, Syn2, Syn3, Syn4, Syn5, Syn6
from .uniform_dataset import DiagDataset, ExpProdUniformDataset, ExpSquaredSumUniformDataset, ExpSquaredSumUniformDatasetV2

list_dataset = {
    "CircleDataset": CircleDataset,
    "SwitchFeature": SwitchFeature,
    "ExpProdGaussianDataset": ExpProdGaussianDataset,
    "ExpSquaredSumGaussianDataset": ExpSquaredSumGaussianDataset,
    "DiagGaussianDataset": DiagGaussianDataset,
    "ExpSquaredSumGaussianDatasetV2": ExpSquaredSumGaussianDatasetV2,
    "S_1": S_1,
    "S_2": S_2,
    "S_3": S_3,
    "Syn1": Syn1,
    "Syn2": Syn2,
    "Syn3": Syn3,
    "Syn4": Syn4,
    "Syn5": Syn5,
    "Syn6": Syn6,
    "DiagDataset": DiagDataset,
    "ExpProdUniformDataset": ExpProdUniformDataset,
    "ExpSquaredSumUniformDataset": ExpSquaredSumUniformDataset,   
    "ExpSquaredSumUniformDatasetV2": ExpSquaredSumUniformDatasetV2,
}