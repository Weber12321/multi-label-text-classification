class PreprocessError(Exception):
    """Error when preprocessing data"""

    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg

    def __reduce__(self):
        return PreprocessError, (self.msg,)

