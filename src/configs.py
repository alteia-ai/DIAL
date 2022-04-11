import attr
import yaml
from icecream import ic


@attr.s(frozen=True)
class MLConfig:
    # Number of classes
    N_CLASSES = attr.ib()
    # Number of input channels (e.g. RGB)
    IN_CHANNELS = attr.ib()
    # Patch size
    WINDOW_SIZE = attr.ib()
    # Name of the architecture
    NET_NAME = attr.ib()
    # pretrain on ImageNet (UNet & LinkNet & SegNet only)
    PRETRAIN = attr.ib()
    # Normalize  data with RGB mean and std
    MEAN_STD = attr.ib()
    # Activate dropout in the network (for bayesian purpose)
    DROPOUT = attr.ib()
    # Train on a subset of data: 1 for full train set, 0 for none of it
    SUB_TRAIN = attr.ib()

    # Train parameters
    EPOCHS = attr.ib()
    OPTIM_BASELR = attr.ib()
    OPTIM_STEPS = attr.ib()
    # Default epoch size is 10 000 samples
    EPOCH_SIZE = attr.ib()
    # Number of threads to use during training
    WORKERS = attr.ib()
    # Number of samples in a mini-batch per GPU
    BATCH_SIZE = attr.ib()
    # Weight the loss for class balancing
    WEIGHTED_LOSS = attr.ib()
    # Data augmentation (flip vertically and horizontally)
    TRANSFORMATION = attr.ib()
    # Heavy augmentation (rotation, shifting,...)
    HEAVY_AUG = attr.ib()
    # Color jittering
    COL_JIT = attr.ib()
    # Keeps x data for validation. If the value is 1,
    # train and test on the full dataset (ie no validation set)
    test_size = attr.ib()
    # Only for interactivity: True enables distance transform to dilate annotations
    DISTANCE_TRANSFORM = attr.ib()
    GUIDED_FILTER = attr.ib()


    # Test parameters
    # stride at test time
    STRIDE = attr.ib()
    # Number of threads to use during testing. Has to be
    # lower than at training since we now load full images.
    TEST_WORKERS = attr.ib()

    # Interactivity
    SAVE_FOLDER = attr.ib()
    PATH_MODELS = attr.ib()

    DISIR = attr.ib()
    DISCA = attr.ib()
    CL_ONLY_NEW_POINTS = attr.ib()
    CL_LR = attr.ib()
    CL_STEPS = attr.ib()
    CL_REG = attr.ib()
    CL_LAMBDA = attr.ib()
    CL_DIST_TRANS = attr.ib()
    ENCODING_SIZE = attr.ib()
    N_CLICKS = attr.ib()  # number of simulated clicks 
    CL_SEQUENTIAL_LEARNING = attr.ib()
    CL_FULL_PRED = attr.ib()  # learn on final full prediction once the annotation process is complete

    #### Active Learning
    BUDGET = attr.ib() # Number of annotations per patch
    AL_DISCA = attr.ib()
    """
    SAMPLING_STRATEGY: Strategy to sample the annotations during evaluation.
        - random (Sample randomly in the mistake areas.)
        - max (Sample the center of the largest mistake areas.)
        - entropy (Sample in the zone maximising the entropy of the prediction in the mistake area)
        - confidnet (Sample in the zone minimizing the confidence of confidnet on the prediction in the mistake area)
        - mcdropout
    """
    SAMPLING_STRATEGY = attr.ib()


def config_factory(config_file: str) -> MLConfig:
    """
    parse the config file and instanciate the config object
    """
    with open(config_file, "rb") as f:
        params = yaml.safe_load(f)
        ic(params)

    return MLConfig(**params)
