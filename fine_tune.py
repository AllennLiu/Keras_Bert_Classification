import numpy as np
import pandas as pd
from json import dumps
from pathlib import Path
from numpy.typing import NDArray
from dataclasses import dataclass
from codecs import open as codecs_open
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Generator, Iterable
from keras import layers, models, optimizers, callbacks
from keras_bert import load_trained_model_from_checkpoint, get_custom_objects, Tokenizer

TRAINED_MODEL_CKPT = './keras_bert.hdf5'
LABEL_CLASS = './label.json'

@dataclass(frozen=True)
class Config:
    max_length : int = 300  # 建议长度 <= 510

@dataclass(frozen=True)
class BertPath:
    config     : str = './chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint : str = './chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict       : str = './chinese_L-12_H-768_A-12/vocab.txt'

class CustomTokenizer(Tokenizer):
    def _tokenize(self, text: str) -> List[str]:
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space 类用未经训练的 [unused1] 表示
            else:
                R.append('[UNK]')      # 剩余的字符是 [UNK]
        return R

def get_token_dict() -> Dict[str, int]:
    """逐行標記語句長度傳入定製 ``Bert Tokenizer`` 作為適用待訓練數據
    的分詞器，最後返回已編碼的字典"""
    token_dict: Dict[str, int] = {}
    with codecs_open(BertPath.dict, encoding='utf-8') as reader:
        for line in reader:
            token_dict[line.strip()] = len(token_dict)
    return token_dict

tokenizer = CustomTokenizer(get_token_dict())

def seq_padding(array: Iterable[int], padding: int = 0) -> NDArray[Any]:
    """
    找出長度最大值後，讓每條文本的長度用 ``padding`` 來填充，使其長度相同

    Parameters
    ----------
    array   : array_like, 文本數組
    padding : int, 填充整數值

    Returns
    -------
    res : ndarray, 長度相同的 ``NumPy Array``
    """
    ML = max([ len(i) for i in array ])
    return np.array([
        np.concatenate([ i, [ padding ] * ( ML - len(i) ) ])
        if len(i) < ML else i for i in array
    ])

class DataGenerator:
    """大量數據先储存在生成器中，節省大量數組元素的內存占用

    Attributes
    ----------
    data: list
        待訓練的文本列表
    batch_size: int
        每次訓練的個數
    shuffle: bool
        文本順序是否打亂
    """
    def __init__(
        self,
        data: List[tuple[str, List[int]]],
        batch_size: int = 8,
        shuffle: bool = True
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Generator[tuple[List[NDArray[Any]], NDArray[Any]], None, None]:
        while True:
            idxs = list(range(len(self.data)))
            self.shuffle and np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:Config.max_length]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [ X1, X2 ], Y
                    [ X1, X2, Y ] = [], [], []

def create_cls_model(num_labels: int) -> models.Model:
    """构建模型
    ------------
    先加載 ``Google`` 預訓練的 ``Bert - chinese_L-12_H-768_A-12`` 模型，取出
    `[CLS]` 对应的向量，后接全连接层，激活函数采用 :class:`~layers.Activation`
    的 `softmax` 函数，完成多分类模型"""
    if Path(args.model).is_file():
        loaded_model: models.Model = models.load_model(args.model, custom_objects=get_custom_objects())
        dense = layers.Dense(num_labels, activation='softmax')(loaded_model.layers[-2].output)
        model = models.Model(loaded_model.input, dense)
    else:
        bert_model = load_trained_model_from_checkpoint(BertPath.config, BertPath.checkpoint, seq_len=None)

        for layer in bert_model.layers:
            layer.trainable = True

        cls_layer = layers.Lambda(lambda x: x[:, 0])(bert_model.output) # 取出 [CLS] 对应的向量用来做分类
        p = layers.Dense(num_labels, activation='softmax')(cls_layer)   # 多分类
        model = models.Model(bert_model.input, p)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(1e-5), # 用足够小的学习率
        metrics=[ 'accuracy' ])
    model.summary()
    return model

def data_labeling(df: pd.DataFrame, labels: NDArray[Any]) -> List[tuple[str, List[int]]]:
    """轉換訓練、測試等數據，標籤化為模型輸入格式"""
    data_list: List[tuple[str, List[int]]] = []
    for i in range(df.shape[0]):
        label, content = df.iloc[i, :]
        label_ids = [ 0 ] * len(labels)
        for j, k in enumerate(labels):
            if k == label:
                label_ids[j] = 1
        data_list.append(( content, label_ids ))
    return data_list

def main() -> None:
    # 数据处理, 读取训练集和测试集
    print('Begin data processing...')
    train_df: pd.DataFrame = pd.read_csv(f'{args.dataset}/train.csv').fillna(value='')
    test_df: pd.DataFrame = pd.read_csv(f'{args.dataset}/test.csv').fillna(value='')

    labels: NDArray[Any] = train_df.label.unique()
    Path(LABEL_CLASS).write_text(
        dumps(dict(zip(range(len(labels)), labels)), ensure_ascii=False, indent=4),
        encoding='utf-8')

    train_data = data_labeling(train_df, labels)
    test_data = data_labeling(test_df, labels)
    train_D = DataGenerator(train_data, args.batch)
    test_D = DataGenerator(test_data, args.batch)
    print('Finish data processing!')

    # 模型训练
    model = create_cls_model(len(labels))
    # 启用对抗训练FGM
    # adversarial_training(model, 'Embedding-Token', 0.5)

    print('Begin model training...')
    # 早停法，防止學習過擬合
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001)
    # 當評價指標不再提升時，減少學習率
    plateau = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', verbose=1, mode='max', factor=0.5, patience=2)
    # 每次訓練完就保存最好的模型
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=args.model, monitor='val_accuracy', mode='max', save_best_only=True)
    model.fit(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=args.epochs,
        validation_data=test_D.__iter__(),
        validation_steps=len(test_D),
        callbacks=[ early_stopping, plateau, model_checkpoint ])
    print('Finish model training!')

    loss, accuracy = model.evaluate(test_D.__iter__(), steps=len(test_D))
    print(f'Loss: {loss * 100:.2f}%, Accuracy: {accuracy * 100:.2f}%')

def args_parser() -> Namespace:
    parser = ArgumentParser(description='Keras to fine tune BERT model implement text classification')
    parser.add_argument('-e', '--epochs', action='store', type=int, default=3,
                        help='the number of training epochs (default: %(default)s)')
    parser.add_argument('-b', '--batch', action='store', type=int, default=16,
                        help='the number of batch size (default: %(default)s)')
    parser.add_argument('-d', '--dataset', action='store', type=str, default='./data/thucnews',
                        help='dataset path (default: %(default)s)')
    parser.add_argument('--model', action='store', type=str, default=TRAINED_MODEL_CKPT,
                        help='trained model file (default: %(default)s)')
    return parser.parse_args()

if __name__ == '__main__':
    args = args_parser()
    main()
