BASE_COLUMNS = ["model", "num_correct"]
ENTROPY_METRICS_OLD = [
    "shannon",
    "condi",
    "joint",
    "renyi",
    "tsallis",
    "diff",
]
ENTROPY_METRICS_NEW = [
    "shannon",
    "m_shannon",
    "condi",
    "m_condi",
    "joint",
    "m_joint",
    "renyi",
    "m_renyi",
    "tsallis",
    "m_tsallis",
    "diff",
]
# Additional columns
ADDITIONAL_COLUMNS_OLD = [
    "head",
    "relation",
    "tail",
    "Result",
]
ADDITIONAL_COLUMNS_NEW = ADDITIONAL_COLUMNS_OLD


# Methods to create column lists
def create_column_list(base, metrics, additional=None):
    columns = base.copy()
    for metric in metrics:
        columns.extend([f"n_{metric}", f"c_{metric}", f"cs_{metric}"])
    if additional:
        columns.extend(additional)
    return columns


def create_column_list_new_version(base, metrics, additional=None):
    columns = base.copy()
    for metric in metrics:
        if "m_" not in metric:
            columns.extend([f"n_{metric}", f"c_{metric}", f"cs_{metric}"])
        else:
            columns.append(f"{metric}")
    if additional:
        columns.extend(additional)
    return columns


def create_column_dtypes(columns):
    dtypes = {col: "float" for col in columns if "m_" not in col}
    dtypes.update(
        {
            "model": "str",
            "relation": "int",
            "num_correct": "int",
            "num_ori": "int",
        }
    )
    dtypes["MRank"] = "object"
    return dtypes


# Creating column lists
column_List = create_column_list(
    BASE_COLUMNS, ENTROPY_METRICS_OLD, ADDITIONAL_COLUMNS_OLD
)
column_List_new = create_column_list_new_version(
    BASE_COLUMNS, ENTROPY_METRICS_NEW, ADDITIONAL_COLUMNS_NEW
)
column_List_eval = create_column_list(BASE_COLUMNS, ENTROPY_METRICS_OLD)
column_List_eval_new = create_column_list_new_version(BASE_COLUMNS, ENTROPY_METRICS_NEW)
column_List_entropy = create_column_list([], ENTROPY_METRICS_OLD)
column_List_entropy_new = create_column_list_new_version([], ENTROPY_METRICS_NEW)

# Creating column data types
entropy_column_dtypes = create_column_dtypes(column_List)
entropy_column_dtypes_new = create_column_dtypes(column_List_new)

# Paths and filename suffixes
default_entropy_dir_path = "./csv/1001/entropy_k_"
default_entropy_strs = [
    "_head_batch.csv",
    "_head_batch_paired.csv",
    "_tail_batch.csv",
    "_tail_batch_paired.csv",
    "_mixed_batch.csv",
    "_mixed_batch_paired.csv",
]
default_entropy_base_strs = [
    "_Tag.csv",
    "_Tag_paired.csv",
]

normal_max = 5
normal_min = 0

entropy_normal_max = 5.0
entropy_normal_min = 2.5

dataset = "FB15K237"
entropy_df = None
strModels = ["TransE", "TransH", "TransR", "TransD"]
# strModels = ["TransE", "TransD"]
hit_k_limits = [10, 50, 100, 250, 1000, 10000, 100000]
rel_stopper_index = 40000

num_count_threshold = 0

types_of_entropy = "n_shannon"

entropy_path_id = None

entropy_path_id_short = None

reverse_flag = True

devices = "cuda:0"

date = "1012"


hold_index = None
debug_flag = False

Mode_Calculator = "Write"
# Mode_Calculator = "Read"
