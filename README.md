# hanging_point_cnn
<img src="https://user-images.githubusercontent.com/39142679/102420461-8ac4b380-4045-11eb-80d8-9848e63ea376.png" width="800">

## setup
```
pip install -e .
```

## Create training dataset
### Rendering
Generate hanging points using [hanging_points_generator](https://github.com/kosuke55/hanging_points_generator)
If you use ycb to generate hanging points
`python run_many.py 'python generate_hanging_points.py'`
you can get contact_points.json like
`<path to ycb urdf> /019_pitcher_base/contact_points/pocky-2020-08-14-18-23-50-720607-41932/contact_points.json`


Next, load the generated contact points and generate training data by rendering on the simulator.
```
cd hangning_points_cnn/create_dataset
python run_many.py 'python renderer.py -n 1000 -i <path to ycb urdf> -s <save dir>'
```
For example,
Set \<save dir\> to `'/media/kosuke/SANDISK-2/meshdata/ycb_hanging_object/rendering'`

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
```
ipython -i -- infer_pose_from_data.py -i <save dir>/<fancy dir> --idx 10 -p <your trained model path>
```
\<fancy dir\>  is made by [eos](https://github.com/iory/eos)
