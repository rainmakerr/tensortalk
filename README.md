## Image captioning with tensorflow

To train your own model, download ms coco dataset from http://mscoco.org/dataset/#download. You need to download 2014 training images and 2014 validation images. Unpack data into ~/.tensortalk/coco and run
```
python prepare_coco_features.py
python train.py
```

Training process can take a few hours. For those who are not willing to wait, pretrained model is available at https://www.dropbox.com/s/2v3iea63nb7dwc5/model.ckpt?dl=0. Now you can caption your images with
```
python generate_captions.py image.jpg [image2.jpg image3.jpg] --model model.ckpt
```

To evaluate your model BLEU score on validation images, run
```
python evaluate.py
```

If you could make the code run on GPU, let me know. Good luck!