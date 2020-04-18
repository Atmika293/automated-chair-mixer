To execute, run the following command : python train_model.py --chooseepoch checkpoint.pth --mode test
For testing the resnet model, our generated test chairs have been converted to grayscale and are in evaluate_chairs/
For training, we generated our own dataset of good and bad chairs which can be found here:

https://drive.google.com/file/d/1LnSRqf_ncwMkB9_CWVbmxp2FOfslK0QN/view

After downloading, place both renders/ and renders_bad/ in chairs-data/ and run this command to train:
python train_model.py --nepoch 25 --mode train