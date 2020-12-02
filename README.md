# classify-image

~~Google Colab~~上で解析を行えるようにした、VGG16など識別系のルーチーンアプリケーションです

## 簡単な使い方の説明

1. 「画像ファイルの入ったフォルダ」と
1. 「各画像ファイルのラベル情報が書かれたcsvファイル」をこれらのコードと同じディレクトリに置き、
1. 「学習のさせ方を記述したoptions.confファイル」を編集して
1. `python image_*.py`を実行する

## 動作保障環境

- Windows 10
- Anaconda 4.8.5
- Python 3.7

## ライブラリのインストール

```None
conda create env -n [env_name] -f environment.yml
```
