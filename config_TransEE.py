column_List = [
    "model",
    "head",
    "num_correct",
    "relation",
    "tail",
    "Result",
    "n_shannon",
    "c_shannon",
    # "n_condi",
    # "c_condi",
    # "n_joint",
    # "c_joint",
    "n_renyi",
    "c_renyi",
    "n_tsallis",
    "c_tsallis",
    "n_diff",
    "c_diff",
]

column_List_new = [
    "model",
    "head",
    "num_correct",
    "relation",
    "tail",
    "Result",
    "n_shannon",
    "c_shannon",
    "m_shannon",
    # "n_condi",
    # "c_condi",
    # "m_condi",
    # "n_joint",
    # "c_joint",
    # "m_joint",
    "n_renyi",
    "c_renyi",
    "m_renyi",
    "n_tsallis",
    "c_tsallis",
    "m_tsallis",
    "n_diff",
    "c_diff",
    "m_diff",
]

column_List_eval = [
    "model",
    "num_correct",
    "n_shannon",
    "c_shannon",
    # "n_condi",
    # "c_condi",
    # "n_joint",
    # "c_joint",
    "n_renyi",
    "c_renyi",
    "n_tsallis",
    "c_tsallis",
    "n_diff",
    "c_diff",
]

column_List_eval_new = [
    "model",
    "num_correct",
    "n_shannon",
    "c_shannon",
    "m_shannon",
    # "n_condi",
    # "c_condi",
    # "m_condi",
    # "n_joint",
    # "c_joint",
    # "m_joint",
    "n_renyi",
    "c_renyi",
    "m_renyi",
    "n_tsallis",
    "c_tsallis",
    "m_tsallis",
    "n_diff",
    "c_diff",
    "m_diff",
]

column_List_entropy = [
    "n_shannon",
    "c_shannon",
    # "n_condi",
    # "c_condi",
    # "n_joint",
    # "c_joint",
    "n_renyi",
    "c_renyi",
    "n_tsallis",
    "c_tsallis",
    "n_diff",
    "c_diff",
]

column_List_entropy_new = [
    "n_shannon",
    "c_shannon",
    "m_shannon",
    # "n_condi",
    # "c_condi",
    # "m_condi",
    # "n_joint",
    # "c_joint",
    # "m_joint",
    "n_renyi",
    "c_renyi",
    "m_renyi",
    "n_tsallis",
    "c_tsallis",
    "m_tsallis",
    "n_diff",
    "c_diff",
    "m_diff",
]

column_List_entropy_new_sort = [
    "c_renyi",
    "c_shannon",
    # "c_condi",
    # "c_joint",
    "c_tsallis",
    "c_diff",
    "n_shannon",
    # "n_condi",
    # "n_joint",
    "n_renyi",
    "n_tsallis",
    "n_diff",
    "m_shannon",
    # "m_condi",
    # "m_joint",
    "m_renyi",
    "m_tsallis",
    "m_diff",
]

column_List_entropy_NM = [
    "c_renyi",
    "n_shannon",
    "c_shannon",
    "c_tsallis",
    "c_diff",
    "n_renyi",
    "n_tsallis",
    "n_diff",
]
column_List_entropy_MINIMUM = [
    "n_shannon",
    "c_shannon",
    "n_renyi",
    "c_renyi",
]
column_List_entropy_MIN = [
    "c_renyi",
    "c_shannon",
    # "c_condi",
    # "c_joint",
    # "c_tsallis",
    # "c_diff",
    "n_shannon",
    # "n_condi",
    # "n_joint",
    "n_renyi",
    # "n_tsallis",
    # "n_diff",
]


column_List_entropy_short = [
    "m_shannon",
    # "m_condi",
    # "m_joint",
    "m_renyi",
    "m_tsallis",
    "m_diff",
    "n_diff",
    "c_diff",
]


default_entropy_strs = [
    "_head_batch.csv",
    # "_head_batch_paired.csv",
    "_tail_batch.csv",
    # "_tail_batch_paired.csv",
    # "_mixed_batch.csv",
    # "_mixed_batch_paired.csv",
]

default_entropy_base_strs = [
    "_Tag.csv",
    # "_Tag_paired.csv",
    # "_mixed_batch.csv",
    # "_mixed_batch_oldfomula.csv",
]


entropy_column_dtypes = {
    "model": "str",
    "relation": "int",
    "num_correct": "int",
    "num_ori": "int",
    "n_shannon": "float",
    "c_shannon": "float",
    "n_renyi": "float",
    "c_renyi": "float",
    "n_tsallis": "float",
    "c_tsallis": "float",
    "n_differential": "float",
    "c_differential": "float",
    "n_p": "float",
    "c_p": "float",
    "MRank": "object",
}

# model,relation,num_correct,num_ori,n_shannon,n_renyi,n_tsallis,n_differential,c_differential,c_shannon,c_renyi,c_tsallis,n_p,c_p
# model,relation,num_correct,num_ori,n_shannon,n_renyi,n_tsallis,n_differential,c_differential,c_shannon,c_renyi,c_tsallis,n_p,c_p


entropy_column_dtypes_new = {
    "model": "str",
    "relation": "int",
    "num_correct": "int",
    "num_ori": "int",
    "n_shannon": "float",
    "c_shannon": "float",
    "m_shannon": "float",
    # "n_joint": "float",
    # "c_joint": "float",
    # "m_joint": "float",
    "n_renyi": "float",
    "c_renyi": "float",
    "m_renyi": "float",
    "n_tsallis": "float",
    "c_tsallis": "float",
    "m_tsallis": "float",
    "n_diff": "float",
    "c_diff": "float",
    "m_diff": "float",
    "n_P": "object",
    "c_P": "object",
    "m_P": "object",
}

str_entropy_output_default = [
    "n_shannon",
    "n_renyi",
    "n_tsallis",
    "n_differential",
    "c_differential",
    "c_shannon",
    "c_renyi",
    "c_tsallis",
    "m_differential",
    "m_shannon",
    "m_renyi",
    "m_tsallis",
    "n_p",
    "c_p",
    "m_p",
]

str_entropy_output_no_maha = [
    "n_shannon",
    "n_renyi",
    "n_tsallis",
    "n_differential",
    "c_differential",
    "c_shannon",
    "c_renyi",
    "c_tsallis",
    "n_p",
    "c_p",
]

str_entropy_output_scipy = [
    "n_diff_scipy",
    "c_diff_scipy",
    "m_diff_scipy",
    "n_p",
    "c_p",
    "m_p",
]

str_entropy_output_scipy_no_maha = [
    "n_diff_scipy",
    "c_diff_scipy",
    "n_p",
    "c_p",
]

entropy_output = {
    "DEFAULT": str_entropy_output_default,
    "DEFAULT_NO_MAHA": str_entropy_output_no_maha,
    "SCIPY": str_entropy_output_scipy,
    "SCIPY_NO_MAHA": str_entropy_output_scipy_no_maha,
}


GROUND_SCORE = {
    "TransE": 0,
    "TransH": 0,
    "TransR": 0,
    "TransD": 0,
}

GROUND_RANK = {
    "TransE": 0,
    "TransH": 0,
    "TransR": 0,
    "TransD": 0,
}

GROUND_RANK_DOWN = {
    "TransE": 0,
    "TransH": 0,
    "TransR": 0,
    "TransD": 0,
}

normal_max = 5
normal_min = 1

entropy_normal_max = 10.0
entropy_normal_min = 5.0
# entropy_normal_min = 4.5
# entropy_normal_min = 3.6875
# entropy_normal_min = 3.75
# entropy_normal_min = 1.0

default_entropy_dir_path = None
# default_entropy_dir_path = "./csv/FB15K237/GOOD_PERFOMANCE/PDF_Categorical_Mixed/entropy_k_"

dataset = "FB15K"
# dataset = "FB15K237"
# dataset = "WN18RR"

# data_tag = "/0925"
data_tag = "/Pre_FB15K237"
# data_tag = "/Pre_WN18RR"

entropy_df = None
strModels = ["TransE", "TransH", "TransR", "TransD"]
# strModels = ["TransH", "TransR", "TransD"]
# strModels = ["TransH", "TransD"]
# hit_k_limits = [10, 50, 100, 250, 500]
# hit_k_limits = [10, 50, 100]
# hit_k_limits = [10, 29, 36]
# hit_k_limits = [5, 10]
hit_k_limits = [5]


rel_stopper_index = 400000

num_count_threshold = 0

# num_count_thresholds = [0, 10, 50, 100, 250, 500, 1000]
# num_count_thresholds = [-2, -1, 10, 32, 50, 64, 100, 250, 500]
# num_count_thresholds = [10, 32, 50, 64, 100, 250, 500]
# num_count_thresholds = [10, 36, 50, 100]
# num_count_thresholds = [1000]
# num_count_thresholds = [36, 29, 10]

# num_count_thresholds = [5, 10, -1, -2]
# num_count_thresholds = [5]
# num_count_thresholds = [10, 18, 29, 36]

# num_count_thresholds = [-2, -1, 10, 36, 50, 100]

# -2: choose best rank model
# -1: all weights are 1
# num_count_thresholds = [-2, -1]
num_count_thresholds = [5]
# num_count_thresholds = [-3]
# num_count_thresholds = [-1]

types_of_entropy = "n_shannon"

entropy_path_id = None

entropy_path_id_short = None

reverse_flag = False

devices = "cuda:0"

# date = "1017_Trained"
date = "1017_Valid"
# date = "test"


hold_index = None
debug_flag = False

Mode_Calculator = "Write"
# Mode_Calculator = "Read"

# MODE_DATA = "TRAINING"
# MODE_DATA = "VALIDATE"
MODE_DATA = "MIXED"

bin_weight = 0.15
bins = {"tail_batch": None, "head_batch": None}

ground = {}
entropy_batch = None

probs_enable = False

pd_batch_size = 1000

ENTROPY_RESOURCES = ["euclidean", "cosine"]
# ENTROPY_RESOURCES = ["euclidean", "cosine", "mahalanobis"]


# MODE = "EMPTY_IMAGE_SHOW"
# MODE = "IMAGE_SAVE"
# MODE = "IMAGE_SHOW"
# MODE = "EMPTY_PRINT_ENTROPY_STAY"
# MODE = "RUN"
# MODE = "EMPTY"
MODE = "PRINT_ENTROPY_STAY"
# MODE = "DEBUT_EMPTY_IMAGE_SAVE"
# MODE = "PRINTABLE"
# MODE = "DEBUG_SHOW"
# MODE = "HIST_IMAGE_SHOW_SAVE"

CALC_MODE = "DEFAULT_NO_MAHA"
# CALC_MODE = "SCIPY"

# CURRENT_ENTROPY_SELECTOR = "cdf"
CURRENT_ENTROPY_SELECTOR = "pdf"
# CURRENT_ENTROPY_SELECTOR = "hist"
# CURRENT_ENTROPY_SELECTOR = "curve"


CURRENT_REL = "None"
CURRENT_HIT = "None"
CURRENT_MODEL = "None"
CURRENT_BATCH = "None"
CURRENT_PAIRD = "None"
CURRENT_LABEL = "None"


MAHALANOBIS_SLICE = False

n_bins = {}
c_bins = {}
m_bins = {}

MODE_EVAL_LIST = ["Optimal", "Pre"]

# MODE_EVALUATION = "DROP_TRUE"
MODE_EVALUATION = "DEFAULT"

MODE_EVALUATION_TOP = False
# MODE_EVALUATION_TOP = True

# EVALUATION_ENTROPY_HOLD = False

WRITE_EVAL_RESULT = False
PATH_EVAL_RESULT = "./csv/Eval_result/"

# Original Alpha : 0.5

rounds = 7
# alpha = 0.75
alpha = 0.5
# alpha = 0.25
# q = 1.5
q = 0.75
# q = 0.5

# CALC_ENTROPY_RESOURCE_TYPE = "no_maha"

ctime = None

EVAL_DEFALT_TF = 18

# MODE_EVAL_NORM = "MINMAX"
# MODE_EVAL_NORM = "TB_MINMAX"
# MODE_EVAL_NORM = "SIGMOID"
# MODE_EVAL_NORM = "TB_SIGMOID"
MODE_EVAL_NORM = "LOGIT"

BEFORE_REL = -1

HPO_THRESHOLD = 0.0001


def reset_GROUND():
    global GROUND_SCORE
    global GROUND_RANK
    global GROUND_RANK_DOWN

    GROUND_SCORE = {
        "TransE": 0,
        "TransH": 0,
        "TransR": 0,
        "TransD": 0,
    }

    GROUND_RANK = {
        "TransE": 0,
        "TransH": 0,
        "TransR": 0,
        "TransD": 0,
    }

    GROUND_RANK_DOWN = {
        "TransE": 0,
        "TransH": 0,
        "TransR": 0,
        "TransD": 0,
    }


MODE_HPO_FAST = False
# MODE_HPO = "THRESHOLD_GROUND"
# MODE_HPO = "THRESHOLD"
MODE_HPO = "MINNORM"

NORM_GAIN = 0.5
NORM_GAIN = 1.5


def setDataset(strDataset):
    global dataset
    global data_tag

    dataset = strDataset

    data_tag = f"/Pre_{strDataset}"


MODE_MIN_RESOURCE = 100000


# NEW_AVG = "Pers"
# NEW_AVG = "TotPer"
NEW_AVG = "auto"

# NEW_AVG = "Total"


Start_Index = 0
# Start_Index = 400
# strModels = ["TransR"]

# 5*NWeight
NWeight = 4

MIN_HIT_PAIRED = False


def change_Hit_Mode(strMode):
    global NEW_AVG
    if NEW_AVG == "auto":
        NEW_AVG = strMode


dist_Euclidean = None
dist_Cosine = None
dist_Tag = "Euclidean"
