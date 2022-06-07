from .loader_artificial import LoaderArtificial
from .circle_dataset import CircleDataset
from .gaussian_dataset import SwitchFeature, create_toeplitz_covariance, create_blockwise_covariance, ExpProdGaussianDataset, ExpSquaredSumGaussianDataset, DiagGaussianDataset, ExpSquaredSumGaussianDatasetV2
from .realx_dataset import S_1, S_2, S_3
from .realx_dataset_same_codeL2x import Syn1, Syn2, Syn3, Syn4, Syn5, Syn6
from .uniform_dataset import DiagDataset, ExpProdUniformDataset, ExpSquaredSumUniformDataset, ExpSquaredSumUniformDatasetV2