class PreprocessError(Exception):
    """Error when preprocessing data"""
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg

    def __reduce__(self):
        return PreprocessError, (self.msg,)


class ModelNotFoundError(Exception):
    """model is not initialized before training"""
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg

    def __reduce__(self):
        return ModelNotFoundError, (self.msg,)


class BackendNotFoundError(Exception):
    """backend is not in the inference configuration"""
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg

    def __reduce__(self):
        return BackendNotFoundError, (self.msg,)
