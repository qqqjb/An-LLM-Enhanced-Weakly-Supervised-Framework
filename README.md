# An-LLM-Enhanced-Weakly-Supervised-Framework

This is the official PyTorch implementation of our paper:
**"Enhancing Weakly Supervised Video Anomaly Detection via Spatiotemporal Context Modeling and Multi-Perspective Semantic Label Generation"**


## Highlights

- We present a novel weakly supervised VAD method. By incorporating SCAT modeling for spatiotemporal feature enhancement and Multi-Perspective Semantic Label Generation (MPSLG) for rich textual representations, the framework significantly improves detection in surveillance videos while addressing background noise and computational efficiency through cross-modal fusion. Specifically, SCAT captures long- and short-term temporal dependencies via hierarchical windowed attention while suppressing environmental noise through context-aware feature aggregation (CAFA), and MPSLG generates semantically rich labels from multiple perspectives via LLM-based prompt engineering, enhancing both interpretability and discrimination capability for accurate anomaly localization.

- The proposed SCAT effectively captures spatiotemporal patterns via a Swin-Transformer and context-aware attention pooling (CAP), optimizing feature weights to correlate video content with anomalous behaviors and suppress environmental noise interference.

- The proposed MPSLG is a strategy leveraging LLMs to generate semantically rich labels from multiple perspectives via prompt templates. This enhances model interpretability and suspicious activity identification by providing detailed textual features for cross-modal alignment, addressing label noise and class imbalance in weakly supervised scenarios.

- The proposed method achieves excellent results on public datasets XD-Violence and UCF-Crime, outperforming state-of-the-art methods and validating its effectiveness.

## Training

### Setup
We extract CLIP features for UCF-Crime and XD-Violence datasets from [VadCLIP](https://github.com/nwpu-zxr/VadCLIP). Download these features using the links below:

| Benchmark | CLIP]    | Model | 
|--------|----------|-----------|
| UCF-Crime   | [Code: 7yzp](https://pan.baidu.com/s/1OKRIxoLcxt-7RYxWpylgLQ) | [OneDrive](https://1drv.ms/u/c/0cc64fe06001c8e6/IQAGlA9x6LC-R4eELzHRuzQ6ARC6ESkceK8IAu_IhLQBlUI?e=bR4iAH)     |
| XD-Violence | [Code: v8tw](https://pan.baidu.com/s/1q8DiYHcPJtrBQiiJMI7aJw)| [OneDrive](https://1drv.ms/u/c/0cc64fe06001c8e6/IQDv64IsPxzFSoyTBqt2YzyxAUJY0vtgBXcBLv8X8gqb99A?e=70OQQ6)      |

The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the download datasets above in `list/xd_CLIP_rgb.csv` and `list/xd_CLIP_rgbtest.csv`. 
- Feel free to change the hyperparameters in `xd_option.py` and `ucf_option.py`

### Train and Test
After the setup, simply run the following command: 

Training and inference for XD-Violence dataset
```bash
python xd_train.py
python xd_test.py
```

Training and inference for UCF-Crime dataset
```bash
python ucf_train.py
python ucf_test.py
```

## Acknowledgments
We thank the authors of [VadCLIP](https://github.com/nwpu-zxr/VadCLIP) for providing the pre-extracted CLIP features.

## Citation

If you find this repo useful for your research, please consider citing our paper:

```bibtex
@article{scat_mpslg_2025,
  title={Enhancing Weakly Supervised Video Anomaly Detection via Spatiotemporal Context Modeling and Multi-Perspective Semantic Label Generation},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2025}
}
```
```
