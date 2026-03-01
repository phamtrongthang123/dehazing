# TODO

## PICMUS Training & Inference

- [ ] Train PICMUS tissue model: `bash train_run.sh` with `score_picmus_tissue.yaml`
- [ ] Train PICMUS haze model: `bash train_run.sh` with `score_picmus_haze.yaml`
- [ ] Update checkpoint paths in `joint_diffusion/configs/inference/paper/picmus_dehaze_pigdm.yaml` (`run_id.sgm` and `sgm.corruptor_run_id`)
- [ ] Run PICMUS inference and verify B-mode output
