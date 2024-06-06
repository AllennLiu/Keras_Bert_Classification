# Keras BERT Classification

通過 `Keras` 微調 `Bert` 模型，可對文本或代碼進行標籤與分類
項目內以新聞文章分類作為範例，代碼分類部分則是使用多個 `Python`、`Shell` 腳本進行測試，同樣可以實現精準分類的效果！

## 項目結構

```bash
keras-bert-classification/
├── chinese_L-12_H-768_A-12 (BERT 中文預訓練模型)
│   ├── bert_config.json
│   ├── bert_model.ckpt.data-00000-of-00001 (該文件大小 393M 需自行下載)
│   ├── bert_model.ckpt.index
│   ├── bert_model.ckpt.meta
│   └── vocab.txt
├── data (已處理的數據集存放文件夾)
│   └── thucnews
│       ├── test.csv
│       └── train.csv
├── label.json   (類別對應字典，訓練後會生成的文件)
├── evaluate.py  (模型效果評估脚本)
├── fine_tune.py (模型微調訓練腳本)
├── predict.py   (模型預測腳本)
├── README.md
├── requirements.txt
└── thucnews_to_dataset.py (整理 THUCNews 新聞數據為數據集)
```

---

## 數據集準備

目前項目裡預準備小型數據，大型數據還需從外部下載後，再做數據處理。

主要分為訓練集 *(train.csv)* 與測試集 *(test.csv)*，其中訓練集每個分類成 `800` 條樣本，測試集每個分類 `100` 條樣本。

#### `THUCNews` 數據集
下載下來的大型數據集共有 `14` 個類別標籤，分別為：

> 财经、时尚、科技、教育、彩票、星座、家居、游戏、股票、社会、房产、时政、体育、娱乐。

每個類別數據量不是平均分配，有的多有的少，使用 `find THUCNews -name "*.txt" | wc -l` 獲取數據量**共 `836075` 筆**

  - **BERT `chinese_L-12_H-768_A-12`** 預訓練模型下載：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    1. 文件有點大 `394M`，故不放在項目中
    2. 下載下來後直接解壓在**項目文件夾**

    ```bash
    wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    unzip chinese_L-12_H-768_A-12.zip
    ```

  - 本項目已準備好的 **THUCNews** 小型數據集：
    - 訓練數據 `./data/thucnews/train.csv`
    - 測試數據 `./data/thucnews/test.csv`
  - **THUCNews** 大型數據集下載：http://thuctc.thunlp.org/message *(簡單填一下信息就可以獲取下載連結)*
    1. 壓縮包請解壓再**項目文件夾**下
    2. 數據處理方式請執行 `thucnews_to_dataset.py` 腳本

    ```bash
    unzip THUCNews.zip
    python thucnews_to_dataset.py
    ```

---

## 環境配置

1. 請使用版本 `Python 3.9+`。

    ```bash
    python -V
    Python 3.9.11
    ```

2. 安裝 `PyPi` 依賴

    ```bash
    python3 -m pip install -r requirements.txt
    ```

---

## 使用方式

  - **微調**訓練 **THUCNews** 新聞數據集，模型參數：`batch_size = 16, maxlen = 300, epoch = 3`。

    ```bash
    python fine_tune.py -d ./data/thucnews
    ```

  - 訓練好的模型文件大小至少有 `1.2G`，訓練採 `Check Point` 方式保存模型，可隨時再次訓練。
  - 以下可以看到訓練 `3` 次後**準確度**達到 **`99.20%`**

    ```bash
    Epoch 1/3
    250/250 [==============================] - 4757s 19s/step - loss: 0.5359 - accuracy: 0.8460 - val_loss: 0.1088 - val_accuracy: 0.9660 - lr: 1.0000e-05
    Epoch 2/3
    250/250 [==============================] - 4749s 19s/step - loss: 0.1595 - accuracy: 0.9523 - val_loss: 0.0696 - val_accuracy: 0.9820 - lr: 1.0000e-05
    Epoch 3/3
    250/250 [==============================] - 4805s 19s/step - loss: 0.0805 - accuracy: 0.9760 - val_loss: 0.0314 - val_accuracy: 0.9940 - lr: 1.0000e-05
    Finish model training!
    32/32 [==============================] - 185s 6s/step - loss: 0.0359 - accuracy: 0.9920
    Loss: 3.59%, Accuracy: 99.20%
    ```

    |    -     | Epoch1 | Epoch2 | Epoch3 | Average |
    | -------- | ------ | ------ | ------ | ------- |
    | Accuracy | 0.9660 | 0.9820 | 0.9940 | 0.9807  |
    | Loss     | 0.1088 | 0.0696 | 0.0314 | 0.0699  |

  - 最後，使用訓練好的模型，對輸入的一句話進行預測 *(輸入的信息會統一被自動轉換為簡體中文)*

    ```bash
    python predict.py -m '馬上就要到開學季了，家長們紛紛報名了各種才藝班，為了不讓自己的孩子輸在起跑點！'
    ```

    ```bash
    1/1 [==============================] - 7s 7s/step
    原文: 马上就要到开学季了，家长们纷纷报名了各种才艺班，为了不让自己的孩子输在起跑点！
    预测标签: 教育
    time taken: 0:00:18.700042
    ```

---

## 模型效果

  - 對測試數據集的每筆數據進行預測，評估模型效果

    ```bash
    python evaluate.py -d ./data/thucnews
    ```

    ```bash
    Predict 1 samples
    1/1 [==============================] - 6s 6s/step
    ...
    ```

  - 依照測試數據集樣本來評估，此處跑了 `500` 個樣本。
  - 評估指标為 `weighted avg F1 score`，可以看到 `f1-score` 幾乎接近 `100%` 的準確度！

    ```bash
    ...
    Predict 500 samples
    1/1 [==============================] - 0s 490ms/step
    Model evaluate result:
                  precision    recall  f1-score   support

              体育     1.0000    1.0000    1.0000        67
              娱乐     1.0000    1.0000    1.0000        64
              家居     1.0000    1.0000    1.0000        17
              彩票     1.0000    1.0000    1.0000         7
              房产     1.0000    1.0000    1.0000        17
              教育     1.0000    1.0000    1.0000        23
              时尚     1.0000    1.0000    1.0000         4
              时政     0.9000    1.0000    0.9474        36
              星座     1.0000    1.0000    1.0000         1
              游戏     1.0000    1.0000    1.0000        10
              社会     1.0000    1.0000    1.0000        30
              科技     1.0000    0.9897    0.9948        97
              股票     1.0000    0.9717    0.9856       106
              财经     1.0000    1.0000    1.0000        21

        accuracy                         0.9920       500
      macro avg     0.9929    0.9972    0.9948       500
    weighted avg     0.9928    0.9920    0.9922       500

    time taken: 0:05:35.097810
    ```

---

## 參考來源

  - https://github.com/percent4/keras_bert_text_classification/tree/master
  - https://github.com/CyberZHG/keras-bert
  - https://blog.csdn.net/hejp_123/article/details/105432539
