# An-LLM-Enhanced-Weakly-Supervised-Framework

Training
Setup
We extract CLIP features for UCF-Crime and XD-Violence datasets, and release these features and pretrained models as follows:

Benchmark	CLIP[Baidu]	CLIP	Model[Baidu]	Model
UCF-Crime	Code: 7yzp	OneDrive	Code: kq5u	OneDrive
XD-Violence	Code: v8tw	OneDrive	Code: apw6	OneDrive
The following files need to be adapted in order to run the code on your own machine:

Change the file paths to the download datasets above in list/xd_CLIP_rgb.csv and list/xd_CLIP_rgbtest.csv.
Feel free to change the hyperparameters in xd_option.py
