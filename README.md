## Progressive Limb-Aware Virtual Try-On, ACM MM'22.
Official code for ACM MM 2022 paper '## Progressive Limb-Aware Virtual Try-On'
[[Checkpoints for test]](The checkpoints are coming soon...) 
[[Paper]](https://dl.acm.org/doi/10.1145/3503161.3547999)

## Environment
python 3.7
torch 1.9.0+cu111
torchvision 0.10.0+cu111

## Dataset
**VITON Dataset** For the dataset, please refer to [VITON](https://github.com/xthan/VITON).

## Inference
```bash
python test.py
```
**Note that** the results of our pretrained model are only guaranteed in VITON dataset only.

## Sample Try-on Results
  
![image](https://github.com/xyhanHIT/PL-VTON/blob/master/images/experiment.png)

## License
The use of this code is restricted to non-commercial research and educational purposes.

## Citation
If our code is helpful to your work, please cite:
```
@inproceedings{han2022progressive,
  title={Progressive Limb-Aware Virtual Try-On},
  author={Han, Xiaoyu and Zhang, Shengping and Liu, Qinglin and Li, Zonglin and Wang, Chenyang},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={2420--2429},
  year={2022}
}
@inproceedings{han2018viton,
  title={VITON: An Image-Based Virtual Try-On Network},
  author={Han, Xintong and Wu, Zuxuan and Wu, Zhe and Yu, Ruichi and Davis, Larry S},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7543--7552},
  year={2018}
}
@inproceedings{wang2018toward,
  title={Toward Characteristic-Preserving Image-Based Virtual Try-On Network},
  author={Wang, Bochao and Zheng, Huabin and Liang, Xiaodan and Chen, Yimin and Lin, Liang and Yang, Meng},
  booktitle={Proceedings of the European Conference on Computer Vision},
  pages={589--604},
  year={2018}
}
@inproceedings{minar2020cp,
  title={CP-VTON+: Clothing Shape and Texture Preserving Image-Based Virtual Try-On},
  author={Minar, Matiur Rahman and Tuan, Thai Thanh and Ahn, Heejune and Rosin, Paul and Lai, Yu-Kun},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2020}
}
@inproceedings{yang2020towards,
  title={Towards Photo-Realistic Virtual Try-On by Adaptively Generating-Preserving Image Content},
  author={Yang, Han and Zhang, Ruimao and Guo, Xiaobao and Liu, Wei and Zuo, Wangmeng and Luo, Ping},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7850--7859},
  year={2020}
}
@inproceedings{ge2021parser,
  title={Parser-Free Virtual Try-On via Distilling Appearance Flows},
  author={Ge, Yuying and Song, Yibing and Zhang, Ruimao and Ge, Chongjian and Liu, Wei and Luo, Ping},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={8485--8493},
  year={2021}
}
```