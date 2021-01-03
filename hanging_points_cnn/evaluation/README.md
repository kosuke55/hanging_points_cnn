## an example for pouring

1. Annotate objects using [pose_annotation_tool](https://github.com/kosuke55/pose_annotation_tool) after making an annotation directory with [make_annotation_dir.py](https://github.com/kosuke55/pose_annotation_tool/blob/master/utils/make_annotation_dir.py)

```
python make_coords_json.py -i '/media/kosuke55/SANDISK/meshdata/ycb_pouring_object_16/textured_urdf'
rosrun annotation_tool annotation_tool
```
2. Convert manual annotation format to coords json using [make_coords_json.py](https://github.com/kosuke55/pose_annotation_tool/blob/master/utils/make_coords_json.py).
```
python make_coords_json.py -i '/media/kosuke55/SANDISK/meshdata/ycb_pouring_object_16/textured_urdf/annotation_obj'
```

3. Create evaluation data with [renderer_create_eval.py](hanging_points_cnn/create_datase/renderer_create_eval.py)
```
python renderer_create_eval.py -i /media/kosuke55/SANDISK/meshdata/ycb_pouring_object_16/textured_urdf -a /media/kosuke55/SANDISK/meshdata/ycb_pouring_object_16/textured_urdf/annotation_obj --task pouring
```

4. Calculate error with manual annotation.
```
python eval_ycb.py -i /media/kosuke55/SANDISK-2/meshdata/ycb_eval -p  <pretrained model>
```
