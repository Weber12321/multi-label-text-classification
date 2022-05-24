from datetime import datetime
from decimal import Decimal
from typing import Dict, Any


def orm_cls_to_dict(record) -> Dict[str, Any]:
    result_dict = {}
    for c in record.__table__.columns:
        if isinstance(getattr(record, c.dataset_name), datetime):
            result_dict[c.dataset_name] = getattr(record, c.dataset_name).strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(getattr(record, c.dataset_name), Decimal):
            result_dict[c.dataset_name] = float(getattr(record, c.dataset_name))
        else:
            result_dict[c.dataset_name] = getattr(record, c.dataset_name)
    return result_dict
