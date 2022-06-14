from utils.preprocess_helper import read_rule_json
if __name__ == '__main__':
    data = read_rule_json('rule_data.json')
    for k, v in data.items():
        print(k)
        print(len(v))

