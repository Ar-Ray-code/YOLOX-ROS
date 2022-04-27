# Person-Detection-using-RaspberryPi-CPU
Raspberry Pi 4のCPU動作を想定した人検出モデルとデモスクリプトです。<br>

https://user-images.githubusercontent.com/37477845/165421632-600f5f63-51e5-4afa-a0d5-3abc59d0d711.mp4

PINTOさんの「[TensorflowLite-bin](https://github.com/PINTO0309/TensorflowLite-bin)」を使用し4スレッド動作時で45~60ms程度で動作します ※1スレッドは75ms前後<br>
ノートPC等でも動作しますが、精度が必要であれば本リポジトリ以外の物体検出モデルをおすすめします。<br>
また、ノートPC使用時は「model.onnx」のほうが高速なケースが多いです。※Core i7-8750Hで10ms前後

# Requirement 
opencv-python 4.5.3.56 or later<br>
tensorflow 2.8.0 or later ※[TensorflowLite-bin](https://github.com/PINTO0309/TensorflowLite-bin)の使用を推奨<br>
onnxruntime 1.9.0 or later ※model.onnxを使用する場合のみ

# Demo
デモの実行方法は以下です。
```bash
python demo.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：640
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：360
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/model.tflite
* --score_th<br>
検出閾値<br>
デフォルト：0.4
* --nms_th<br>
NMSの閾値<br>
デフォルト：0.5
* --num_threads<br>
使用スレッド数 ※TensorFlow-Lite使用時のみ有効<br>
デフォルト：None

# Demo(ROS2)
ROS2向けのデモです。

ターミナル1
```bash
ros2 run v4l2_camera v4l2_camera_node
```

ターミナル2
```bash
python3 ./demo_ros2.py
```

# Reference
* [PINTO0309/TensorflowLite-bin](https://github.com/PINTO0309/TensorflowLite-bin)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Person-Detection-using-RaspberryPi-CPU is under [Apache 2.0 License](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[イギリス・ロンドン　リージェント・ストリート](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002160979_00000)を使用しています。
