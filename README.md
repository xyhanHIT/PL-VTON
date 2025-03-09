## Progressive Limb-Aware Virtual Try-On, ACM MM'22.
Official code for ACM MM 2022 paper 'Progressive Limb-Aware Virtual Try-On'

We propose a novel progressive limb-aware virtual try-on framework named PL-VTON. PL-VTON consists of Multi-attribute Clothing Warping (MCW), Human Parsing Estimator (HPE), and Limb-aware Texture Fusion (LTF), which produces stable clothing deformation and handles the texture retention well in the final try-on result.

[[Paper]](https://dl.acm.org/doi/10.1145/3503161.3547999)

[[Checkpoints]](https://drive.google.com/file/d/18KvqkWWbjI_GHkqF5HZes0RNB233DHPG/view?usp=share_link)

## Notice
IEEE Transactions on Multimedia 2023 paper by us (follow-up research): https://github.com/xyhanHIT/PL-VTONv2

## Pipeline
![image](https://github.com/xyhanHIT/PL-VTON/blob/master/images/pipeline.png)

## Environment
python 3.7

torch 1.9.0+cu111

torchvision 0.10.0+cu111

## Dataset
For the dataset, please refer to [VITON](https://github.com/xthan/VITON).

## Inference
1. Download the checkpoints from [here](https://drive.google.com/file/d/18KvqkWWbjI_GHkqF5HZes0RNB233DHPG/view?usp=share_link).

2. Get [VITON dataset](https://github.com/xthan/VITON).

3. Run the "test.py".
```bash
python test.py
```
**Note that** the results of our pretrained model are guaranteed in VITON dataset only.

## License
The use of this code is restricted to non-commercial research and educational purposes.

## Citation
If you use our code or models, please cite with:
```bibtex
@inproceedings{han2022progressive,
  title={Progressive Limb-Aware Virtual Try-On},
  author={Han, Xiaoyu and Zhang, Shengping and Liu, Qinglin and Li, Zonglin and Wang, Chenyang},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={2420--2429},
  year={2022}
}
```
