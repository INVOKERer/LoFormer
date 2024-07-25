

# LoFormer:  (ACM MM 24)
[Xintian Mao](https://scholar.google.es/citations?user=eM5Ogs8AAAAJ&hl=en), Jiansheng Wang, Xingran Xie , [Qingli Li] and [Wang Yan](https://scholar.google.com/citations?user=5a1Cmk0AAAAJ&hl=en)




## Quick Run

## Training
1. Download GoPro training and testing data
2. To train the LoFormer, run
 ```
cd LoFormer
./train_8gpu.sh Motion_Deblurring/Options/train_GoPro_LoformerLarge_600k_8gpu.yml
```


## Evaluation
To test the pre-trained models of Deblur [百度网盘]()(提取码:) on your own images, run 
```
python Motion_Deblurring/val.py 
```


## Results
Results on GoPro, HIDE, Realblur test sets:
[百度网盘]()(提取码:)

## Reference Code:
- https://github.com/megvii-research/NAFNet
- https://github.com/INVOKERer/DeepRFT/tree/AAAI2023

## Citation
If you use , please consider citing:
```

```
## Contact
If you have any question, please contact mxt_invoker1997@163.com

**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox. 

## Our Related Works
- Deep Residual Fourier Transformation for Single Image Deblurring, arXiv 2021. [Paper](https://arxiv.org/abs/2111.11745) | [Code](https://github.com/INVOKERer/DeepRFT)
- Intriguing Findings of Frequency Selection for Image Deblurring, AAAI 2023. [Paper](https://arxiv.org/abs/2111.11745) | [Code](https://github.com/DeepMed-Lab-ECNU/DeepRFT-AAAI2023)
