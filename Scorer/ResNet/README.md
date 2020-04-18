The final models for all the seven views can be downloaded from here:
https://drive.google.com/drive/folders/1cVk2kTtZOhH9DgbnSJMOzLf1771dTsn1?usp=sharing

For testing the resnet model, our generated test chairs have been converted to grayscale and can be downloaded here:
https://drive.google.com/drive/folders/1sSwm_FxGuklnl8OArAmtBO-AJEIWIbZ6?usp=sharing
To execute, run the following command : python train_model.py --chooseepoch checkpoint.pth --mode test


For training, we generated our own dataset of good and bad chairs which can be found here:
https://drive.google.com/file/d/1LnSRqf_ncwMkB9_CWVbmxp2FOfslK0QN/view
After downloading, place both renders/ and renders_bad/ in chairs-data/ and run this command to train:
python train_model.py --nepoch 25 --mode train