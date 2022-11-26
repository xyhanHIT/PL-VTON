## Progressive Limb-Aware Virtual Try-On, ACM MM'22.
[[Checkpoints]](The checkpoints are coming soon...) 
[[Paper]](https://dl.acm.org/doi/10.1145/3503161.3547999)

## Update
- [2021-10-22] The light point artifacts would occur in current training results. This may be due to some version differences of our training codes when we rearranged them since we didn't observe same artifacts in our released checkpoints. It might be caused by the instablity in training the preservation (**identical mapping**) of clothes region in **Content Fusion Module**. Try to paste back the ground-truth clothes to the CFM results when calculating the VGG loss, Gan loss, Feature Matching loss (**All except L1**), since the above loss might degenerate the results when learning **identical mapping**. L1 loss can be applied to the reconstruction of clothes region to learn this identical mapping. This [ISSUE](https://github.com/switchablenorms/DeepFashion_Try_On/issues/21) addressed this problem.

## Dataset
**VITON Dataset** For the dataset, please refer to [VITON](https://github.com/xthan/VITON).

## Inference
```bash
python test.py
```
**Note that** the results of our pretrained model are only guaranteed in VITON dataset only, you should re-train the pipeline to get good results in other datasets.

## Sample Try-on Results
  
![image](https://github.com/xyhanHIT/PL-VTON/blob/master/images/experiment.png)


## Citation
If you use our code or models, please cite with:
```
@inproceedings{han2022progressive,
  title={Progressive Limb-Aware Virtual Try-On},
  author={Han, Xiaoyu and Zhang, Shengping and Liu, Qinglin and Li, Zonglin and Wang, Chenyang},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={2420--2429},
  year={2022}
}
```