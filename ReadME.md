# Code for GANorCON

```
pip3 install requirements.txt
```

## Training

Place the DatasetGAN released data at ./DatasetGAN_data inside this folder

```bash

python3 eval_face_seg.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path MoCoV2_512_CelebA.pth --learning_rate 0.001 --weight_decay 0.0005 --adam --epochs 800 --cosine --batch_size 2 --log_path ./log.txt --model_name Nvidia_segmentor --model_path ./512_faces_celeba --image_crop 0 --image_size 512 --use_hypercol


```

## Testing

```bash

python3 gen_score_seg.py --resume ./512_faces_celeba/Nvidia_segmentor/resnet50.pth 

```
