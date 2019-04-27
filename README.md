# classify-image
Google Colab上で解析を行えるようにした、VGG16など識別系のルーチーンアプリケーションです

## リポジトリ内のコードの説明について
コードの説明に関しては[Wiki](https://github.com/hiroki-mas-med/classify-image/wiki)を参照してください。

## 簡単な使い方の説明
imgフォルダと同階層にこれらのコードを置いてください。  
imgフォルダの中身はgrade00、grade01みたいに順番に並んでいる画像フォルダが何個かある。  
grade00の中には画像フォルダがいっぱい入っている。そんなイメージです。  
（詳しくは上述の[Wiki](https://github.com/hiroki-mas-med/classify-image/wiki)を参照のこと）

## TODO

### 今後すること
- コーディング規約flake8に準拠する
- github連携してリポジトリにボタンをつける
- salckをgithubに連携してpushしてCIでpassしたら通知するようにする

### コードの中身に関すること
- main.pyのpicmodeのマジックナンバーの解消
- data_augment.py traingn_data.py validation.data.pyを機能が重複しているものを解消
- 画像を作ってから解析ではなくて学習するときにデータを作るようにする
- AUCアナリシスのpushされたらそれに対応して、リファクタリング
- setting.pyに入れる定数の基準が曖昧だから決めよう
- ~~クソ~~多い引数を持つ関数をどうするか

### 議論中
- imgaeフォルダの中にフォルダ分けせずにcsvで画像を辞書形式で管理するか
- kerasの中に左右反転するメソッドがあるから独自に作らなくてもいいのではないか

