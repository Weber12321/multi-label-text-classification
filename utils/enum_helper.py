from enum import Enum


class TrainingStatus(str, Enum):
    training = "training"
    finished = "finished"
    failed = "failed"


class ModelName(str, Enum):
    bert_base = "bert_base"
    XLNet = "XLNet"
    roberta = "roberta"
    albert = "albert"
    XLM_roberta = "XLM_roberta"


class DatasetName(str, Enum):
    go_emotion = "go_emotion"


class DatabaseSelection(str, Enum):
    wh_bbs_01 = "wh_bbs_01"
    wh_blog_01 = "wh_blog_01"
    wh_fb = "wh_fb"
    wh_forum_01 = "wh_forum_01"


class RuleSelection(str, Enum):
    rule_data_v1 = "rule_data_v1"


class TagSelection(str, Enum):
    male = "男性"
    female = "女性"
    employee = "上班族"
    student = "學生"
    parenting = "有子女"
    young = "青年"
    married = "已婚"
    unmarried = "未婚"
