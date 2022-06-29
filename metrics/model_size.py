import pandas as pd


def get_model_size(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()

    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    size_dict = {
        'param_size': param_size,
        'param_sum': param_sum,
        'buffer_size': buffer_size,
        'buffer_sum': buffer_sum,
        'total_size(MB)': size_all_mb
    }
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    size_df = pd.DataFrame(list(size_dict.items()), columns=['name', 'value'])
    size_df = size_df.round(2)
    return size_df
