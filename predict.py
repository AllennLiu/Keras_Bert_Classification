
import numpy as np
from json import loads
from pathlib import Path
from opencc import OpenCC
from time import time as ts
from functools import wraps
from datetime import datetime
from typing import Any, Callable, TypeVar
from keras_bert import get_custom_objects
from keras.models import load_model, Model
from argparse import ArgumentParser, Namespace
from fine_tune import get_token_dict, Config, CustomTokenizer, TRAINED_MODEL_CKPT, LABEL_CLASS

# 预测示例语句
EXAMPLE_TEXT = """说到硬派越野SUV，你会想起哪些车型？是被称为“霸道”的丰田 普拉多 (配置 | 询价) ，还是被叫做“山猫”的帕杰罗，亦或者是“渣男专车”奔驰大G、
“沙漠王子”途乐。总之，随着世界各国越来越重视对环境的保护，那些大排量的越野SUV在不久的将来也会渐渐消失在我们的视线之中，所以与其错过，
不如趁着还年轻，在有生之年里赶紧去入手一台能让你心仪的硬派越野SUV。而今天我想要来跟你们聊的，正是全球公认的十大硬派越野SUV，
越野迷们看完之后也不妨思考一下，到底哪款才是你的菜，下面话不多说，赶紧开始吧。
"""
CC = OpenCC('tw2sp')
CALC_T = TypeVar('CALC_T')

def timeit(func: Callable[..., CALC_T]) -> Callable[..., CALC_T]:
    """
    Calculating Run Time of a function using decorator,
    you could use this to over target function and the
    runtime of a function will output after it done.

    Examples
    -------
    ```
    @timeit
    def test(n):
        return [ i ** i for i in range(n) ]
        x = test(10000)
    ```
    >>> time taken: 0:00:05.531982
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> CALC_T:
        begin_ts = datetime.now()
        result = func(*args, **kwargs)
        print(f'time taken: {datetime.now() - begin_ts}')
        return result
    return wrapper

@timeit
def main() -> None:
    tokenizer = CustomTokenizer(get_token_dict())
    label_dict = loads(Path(LABEL_CLASS).read_text(encoding='utf-8'))

    # 利用 BERT 进行 tokenize 分詞
    text = CC.convert(args.message[:Config.max_length])
    indices, segments = tokenizer.encode(first=text, max_len=Config.max_length)

    # 模型预测并输出预测结果
    model: Model = load_model(args.model, custom_objects=get_custom_objects())
    predicted = model.predict([ np.array([ indices ]), np.array([ segments ]) ])
    result = np.argmax(predicted[0])

    print(f'原文: {text}')
    print(f'预测标签: {label_dict[str(result)]}')

def args_parser() -> Namespace:
    parser = ArgumentParser(description='Using trained model to text classification')
    parser.add_argument('-m', '--message', action='store', type=str, default=EXAMPLE_TEXT,
                        help='which message need to predict (default: %(default)s)')
    parser.add_argument('--model', action='store', type=str, default=TRAINED_MODEL_CKPT,
                        help='trained model file (default: %(default)s)')
    return parser.parse_args()

if __name__ == '__main__':
    args = args_parser()
    main()
