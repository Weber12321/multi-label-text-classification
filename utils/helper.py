from datetime import datetime


def time2string(time_value=datetime.now(), format_string='%Y-%m-%d'):
    return time_value.strftime(format_string)
