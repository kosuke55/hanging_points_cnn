# hanging_point_cnn
<img src="https://user-images.githubusercontent.com/39142679/102420461-8ac4b380-4045-11eb-80d8-9848e63ea376.png" width="800">

## setup
```
pip install -e .
```

## Create training dataset
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
