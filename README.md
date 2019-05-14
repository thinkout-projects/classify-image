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
公開に向けてユーザビリティの向上およびコードの簡素化を行う。

### コードの全体像
![code_image](https://user-images.githubusercontent.com/39452528/57676474-8a8d5680-765f-11e9-8a2f-23ce3a7f6461.png)
上図のような構成に変更する。

### 各コードの修正内容
- image_classifier.py
  - [x] main.pyから分類部分だけを記述する
  - [ ] csvファイルからのデータ読み込みに対応する
  - [ ] エポック数や学習率などのパラメータはコマンドライン引数から設定
- image_regressor.py
  - [x] main.pyから回帰部分だけを記述する
  - [ ] csvファイルからのデータ読み込みに対応する
  - [ ] エポック数や学習率などのパラメータはコマンドライン引数から設定
- learning.py
  - [ ] modelの定義を削除する
  - [ ] csvファイルからのデータ読み込みに対応する
- models.py
  - [ ] learning.pyのmodelを定義している部分を記述する
  - [ ] modelの出力部分を改良する
- data_generator.py
  - [ ] training.py、validation_data.py、data_augment.pyをまとめて記述する
  - [ ] 画像増幅→学習中に画像増幅をやめて、学習中に画像増幅に一本化する
  - [ ] csvファイルからのデータ読み込みに対応する
- auc_nalysis.py
  - [x] 統計部分の修正
  - [ ] csvファイルからのデータ読み込みに対応する
- k_fold_split.py
  - [ ] ユニークIDの分割部分の修正
  - [ ] csvファイルからのデータ読み込みに対応する
- error.py
  - [ ] プログラムのエラーメッセージやエラーコードなどを記述する
  

### その他全体的な修正内容
- [ ] コーディング規約flake8に準拠する
- [ ] マジックナンバーの解消する
- [ ] 機能が重複しているものを解消する
- [ ] ~~クソ~~多い引数を持つ関数を解消する
- [ ] github連携してリポジトリにボタンをつける
- [x] salckをgithubに連携してpushしてCIでpassしたら通知するようにする


### 議論中
- imgaeフォルダの中にフォルダ分けせずにcsvで画像を辞書形式で管理するか
- kerasの中に左右反転するメソッドがあるから独自に作らなくてもいいのではないか
