Forked from the great Repo https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3.

Steps to run:
1. Follow original repo steps 1-4 to configure environment.
2. Download pre-trained VGG16 from [here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) and place it as `data\imagenet_weights\vgg16.ckpt`. 
3. Download KITTI dataset.
4. Place training, training_label folder under `...\\data\VOCdevkit2007\VOC2007`, run `python modify_annotations_txt.py`, `python kitti_txt_to_xml.py` and `python abc.py` to pre-process KITTI data.
5. Config hyper-perameter file in `...\\config\config.py`, then in `...\\` run `python train.py` to train model
6. `...\\default\voc_2007_trainval\default\vgg16_faster_rcnn_iter_10000.ckpt.meta` is the model, copy it into `...\\output\vgg16\voc_2007_trainval\default`, rename it to `vgg16.ckpt.meta`.
7. Place test images in `data\demo`, change relative code in `demo.py` and run `python demo.py` to test and generate results.

Project done together with another repo https://github.com/MuMuPatrick/kitti_yolo, can find KITTI dataset there.
