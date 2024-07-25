

# [LoFormer: Local Frequency Transformer for Image Deblurring](https://arxiv.org/abs/2407.16993)  (ACM MM 24)
[Xintian Mao](https://scholar.google.es/citations?user=eM5Ogs8AAAAJ&hl=en), Jiansheng Wang, Xingran Xie , Qingli Li and [Wang Yan](https://scholar.google.com/citations?user=5a1Cmk0AAAAJ&hl=en)




## Quick Run

## Training
1. Download GoPro training and testing data
2. To train the LoFormer, run
 ```
cd LoFormer
./train_8gpu.sh Motion_Deblurring/Options/train_GoPro_LoformerLarge_600k_8gpu.yml
```

## Evaluation
To test the pre-trained models of Deblur [百度网盘](https://pan.baidu.com/s/1dFUmGO0d6H0cThyLYFz18g)(提取码:ca3l) on your own images, run 
```
python Motion_Deblurring/val.py 
```

## Results
Results on GoPro, HIDE, Realblur test sets:
[百度网盘](https://pan.baidu.com/s/1Tx6_iQ58u95t-jfken3QZQ)(提取码:rcx5)

## Citation
If you use , please consider citing:
```
@inproceedings{xintm2024LoFormer, 
    title = {LoFormer: Local Frequency Transformer for Image Deblurring},
    author = {Xintian Mao, JIansheng Wang, Xingran Xie, Qingli Li and Yan Wang}, 
    booktitle = {Proc. ACM MM}, 
    year = {2024}
    }
```
## Contact
If you have any question, please contact mxt_invoker1997@163.com

## Our Related Works
- Deep Residual Fourier Transformation for Single Image Deblurring, arXiv 2021. [Paper](https://arxiv.org/abs/2111.11745) | [Code](https://github.com/INVOKERer/DeepRFT)
- Intriguing Findings of Frequency Selection for Image Deblurring, AAAI 2023. [Paper](https://arxiv.org/abs/2111.11745) | [Code](https://github.com/DeepMed-Lab-ECNU/DeepRFT-AAAI2023)
- AdaRevD: Adaptive Patch Exiting Reversible Decoder Pushes the Limit of Image Deblurring, CVPR 2024. [Paper](https://arxiv.org/abs/2406.09135) | [Code](https://github.com/INVOKERer/AdaRevD)


## Reference Code:
- https://github.com/megvii-research/NAFNet
- https://github.com/INVOKERer/DeepRFT/tree/AAAI2023
- https://github.com/swz30/Restormer

**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox. 
