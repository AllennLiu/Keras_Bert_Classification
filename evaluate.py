import pandas as pd
from typing import Union
from fine_tune import TRAINED_MODEL_CKPT
from keras_bert import get_custom_objects
from keras.models import load_model, Model
from predict import timeit, predict_single
from argparse import ArgumentParser, Namespace
from sklearn.metrics import classification_report

def evaluate(model: Model, df: pd.DataFrame) -> Union[str, dict]:
    """使用測試數據集的數據來評估模型標籤分類效能"""
    true_y_list, pred_y_list = [], []
    for i in range(df.shape[0]):
        print(f'Predict {i + 1} samples')
        true_y, content = df.iloc[i, :]
        pred_y = predict_single(model, content)[0]
        true_y_list.append(true_y)
        pred_y_list.append(pred_y)

    return classification_report(true_y_list, pred_y_list, digits=4)

@timeit
def main() -> None:
    model: Model = load_model(args.model, custom_objects=get_custom_objects())
    test_df = pd.read_csv(f'{args.dataset}/test.csv').fillna(value='')
    output_data = evaluate(model, test_df)
    print(f'Model evaluate result:\n{output_data}')

def args_parser() -> Namespace:
    parser = ArgumentParser(description='Evaluating trained model effect')
    parser.add_argument('-d', '--dataset', action='store', type=str, default='./data/thucnews',
                        help='dataset path (default: %(default)s)')
    parser.add_argument('--model', action='store', type=str, default=TRAINED_MODEL_CKPT,
                        help='trained model file (default: %(default)s)')
    return parser.parse_args()

if __name__ == '__main__':
    args = args_parser()
    main()
