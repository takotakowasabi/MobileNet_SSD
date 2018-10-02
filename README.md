# MobileNet_SSD

このリポジトリは、私がサマーインターンにおいて学習した成果をまとめておくためのものになります。学習済みの重みやスクリプトは是非自由に使ってください。

Wikiにも調べた内容を少しまとめています。

使用したデータセットは[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)と[WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)です。

- 成果発表資料
  
サマーインターンの成果や調べた内容、感想などをまとめています。

- original_models

SSDやMobileNetの論文から、独自に作成したMobileNet-SSDのモデルのスクリプトです。

- scripts

公開リポジトリである[ssd_keras](https://github.com/pierluigiferrari/ssd_keras)上で動くように設計したスクリプトです。

- study_log

インターン中に日記のようにつけていた備忘録たちです。

- tensorboard_log

tensor board用のlogファイルです。かなり雑多に混ざっていますが、モデルの参考にしていただければと思います。

---

学習済みの重みは[こちら](https://drive.google.com/drive/folders/1IhSM9zV4o1wLTOg0GMe_Dmf2U9iTmpf_?usp=sharing)になります。

### mobilenet_ssd300_ReLU_face_detection_epoch-116_loss-4.4657_val_loss-4.1590

original_modelsのkeras_mobileNet_ssd_ReLUでWIDER FACEを学習した重み

### mobilenet_ssd300_ReLU6_face_detection_epoch-118_loss-4.4522_val_loss-4.1524

original_modelsのkeras_mobileNet_ssd_ReLU6でWIDER FACEを学習した重み

### opensource_base_mobilenet_ssd300_pascal_07+12_epoch-119_loss-3.2959_val_loss-2.9481.h5

[こちら](https://github.com/tanakataiki/ssd_kerasV2/blob/master/model/ssd300MobileNet.py)のモデルでPascal VOCを学習した重み

### ssd300_face_detection_epoch-116_loss-4.5341_val_loss-4.2570

[こちら](https://github.com/pierluigiferrari/ssd_keras/blob/master/models/keras_ssd300.py)のモデルでWIDER FACEを学習した重み

### ssd512_face_detection_epoch-119_loss-3.1685_val_loss-3.1109

[こちら](https://github.com/pierluigiferrari/ssd_keras/blob/master/models/keras_ssd512.py)のモデルでWIDER FACEを学習した重み
