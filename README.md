# hanging_point_cnn
<img src="https://user-images.githubusercontent.com/39142679/102420461-8ac4b380-4045-11eb-80d8-9848e63ea376.png" width="800">

## setup
```
pip install -e .
```

## Create training dataset
### Rendering
Generate hanging points using [hanging_points_generator](https://github.com/kosuke55/hanging_points_generator).
If you use ycb to generate hanging points
`run-many 'python generate_hanging_points.py'`  
you can get contact_points.json like  
`<path to ycb urdf> /019_pitcher_base/contact_points/pocky-2020-08-14-18-23-50-720607-41932/contact_points.json`

Download [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) to <random_texture_path>. 

Render the training image by loading contact points and textures.  
Can be executed in parallel using [eos run-many](https://github.com/iory/eos/blob/master/eos/run_many.py).  
```
cd hangning_points_cnn/create_dataset
run-many 'python renderer.py -n 200 -i <path to ycb urdf> -s <save dir> --random-texture-path <random_texture_path>' -j 10 -n 10
```

### Check annotated data
Use `visualize-function-points` app.
```
visualize-function-points -h
INFO - 2020-12-28 01:56:36,367 - topics - topicmanager initialized
pybullet build time: Sep 14 2020 02:23:24
usage: visualize-function-points [-h] [--input-dir INPUT_DIR] [--idx IDX]
                                 [--large-axis]

optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR, -i INPUT_DIR
                        input urdf (default: /media/kosuke55/SANDISK-2/meshdat
                        a/ycb_eval/019_pitcher_base/pocky-2020-10-17-06-01-16-
                        481902-45682)
  --idx IDX             data idx (default: 0)
  --large-axis, -la     use large axis as visulaizing marker (default: False)
INFO - 2020-12-28 01:56:39,356 - core - signal_shutdown [atexit]
```

| <img src="https://user-images.githubusercontent.com/39142679/103175749-a767b380-48af-11eb-9feb-e39cb1aeea9d.png" width="300" height="300"> <img src="https://user-images.githubusercontent.com/39142679/103175745-a33b9600-48af-11eb-96fc-0d85e8f77e8c.png" width="300" height="300">
|:--:|
| left: hanging points &ensp; right: pouring points|


## Training
Specify the model config and the save path of the generated data
```
cd hanging_points_cnn/learning_scripts
python train_hpnet.py -g 2 -c config/gray_model.yaml  -bs 16 -dp <save dir>
./start_server.sh
```


## Inference
Use `infer-function-points` app.
```
infer-function-points -h
INFO - 2020-12-29 22:41:01,673 - topics - topicmanager initialized
usage: infer-function-points [-h] [--input-dir INPUT_DIR] [--color COLOR]
                             [--depth DEPTH] [--camera-info CAMERA_INFO]
                             [--pretrained_model PRETRAINED_MODEL]
                             [--predict-depth PREDICT_DEPTH] [--task TASK]

optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR, -i INPUT_DIR
                        input directory (default: None)
  --color COLOR, -c COLOR
                        color image (.png) (default: None)
  --depth DEPTH, -d DEPTH
                        depth image (.npy) (default: None)
  --camera-info CAMERA_INFO, -ci CAMERA_INFO
                        camera info file (.yaml) (default: None)
  --pretrained_model PRETRAINED_MODEL, -p PRETRAINED_MODEL
                        Pretrained models (default: /media/kosuke55/SANDISK-2/
                        meshdata/shapenet_pouring_render/1218_mug_cap_helmet_b
                        owl/hpnet_latestmodel_20201219_0213.pt)
  --predict-depth PREDICT_DEPTH, -pd PREDICT_DEPTH
                        predict-depth (default: 0)
  --task TASK, -t TASK  h(hanging) or p(pouring) (default: h)
INFO - 2020-12-29 22:41:03,442 - core - signal_shutdown [atexit]
```

For multiple data.
```
infer-function-points -i <input directory> -p <trained model> --task <hanging or pouring>
```

For specific data.
```
infer-function-points -c <color.png> -d <depth.npy> -ci <camera_info.yaml> -p <trained model> --task <hanging or pouring>
```

Download ycb rgbd with annotation and trained model.
[ycb_real_eval](https://drive.google.com/file/d/1jGcLZ0vDQBx_rqViCwI6bBqhu5RuOiS9/view?usp=sharing)
[pretrained_model](https://drive.google.com/file/d/1m8qluHL0rUYiaef_0WzrMj0nQ9AFWReX/view?usp=sharing)

## Citation
```
@inproceedings{takeuchi_icra_2021,
 author = {Takeuchi, Kosuke and Yanokura, Iori and Kakiuchi, Yohei and Okada, Kei and Inaba, Masayuki},
 booktitle = {ICRA},
 month = {May},
 title = {Annotation-Free Hanging Point Learning from Random Shape Generation and Physical Function Validation},
 year = {2021},
}
```
