# Model checkpoints

Copy your trained `.pt` files here. Do not upload checkpoints to GitHub if the repository is public.

## 1. Car-parts model

The uploaded `Car_Parts_Segmentation(2).ipynb` reports this checkpoint:

`/content/drive/MyDrive/merged_carparts_models/FINAL_best.pt`

Its notebook output reports `task=segment`, 25 classes, and a size of approximately 22.8 MB. Copy it to:

`models/car_parts.pt`

Class names reported by the notebook:

`back_bumper, back_door, back_glass, back_left_door, back_left_light, back_light, back_right_door, back_right_light, front_bumper, front_door, front_glass, front_left_door, front_left_light, front_light, front_right_door, front_right_light, hood, left_mirror, right_mirror, tailgate, trunk, wheel, front_fender, quarter_panel, running_board`.

## 2. Damage model

The uploaded `Car_Damage_FreeGPU_Inspect_Train_Resume_(3)(4).ipynb` trains a one-class YOLOv8 segmentation model with class label `damage`. Its intended finished deployment path is:

`/content/drive/MyDrive/datasets/car_damage_training_output/best_damage_model.pt`

The notebook output included in the upload ends during epoch 93/100, so it does not prove that the final copy step ran. Check Google Drive for the intended file first. If it is absent, check:

`/content/drive/MyDrive/datasets/car_damage_training_output/checkpoints/best.pt`

The checkpoint callback copies `best.pt` after completed epochs. After confirming the checkpoint loads and validation is acceptable, copy it to:

`models/car_damage.pt`

Do not use `last.pt` just because it is easiest to find: `best.pt` is normally the better inference candidate. Keep the original checkpoints unchanged.
