from datetime import datetime
from decimal import Decimal
from typing import Dict, Any


def orm_cls_to_dict(record) -> Dict[str, Any]:
    result_dict = {}
    for c in record.__table__.columns:
        if isinstance(getattr(record, c.name), datetime):
            result_dict[c.name] = getattr(record, c.name).strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(getattr(record, c.name), Decimal):
            result_dict[c.name] = float(getattr(record, c.name))
        else:
            result_dict[c.name] = getattr(record, c.name)
    return result_dict
