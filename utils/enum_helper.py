from enum import Enum


class TrainingStatus(str, Enum):
    training = "training"
    finished = "finished"
    failed = "failed"


class ModelName(str, Enum):
    albert = "albert-base-v2"
    bert_base = "bert-base-uncased"
    bert_chinese = "bert-base-chinese"
    chinese_bert_wwm_ext = "hfl/chinese-bert-wwm-ext"
    chinese_macbert_base = "hfl/chinese-macbert-base"
    chinese_roberta_wwm_ext = "hfl/chinese-roberta-wwm-ext"
    xlm_roberta = "xlm-roberta-base"


class DatasetName(str, Enum):
    go_emotion = "go_emotion"
    audience_tiny = "audience_tiny"


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
