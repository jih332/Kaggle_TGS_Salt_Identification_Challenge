# Kaggle TGS Salt Identification Challenge(2018)  
This is a part of my Kaggle TGS Salt Identification Challenge(2018) solutions. Private LB 230th.  
  
### Deep Learning Framework:  
Pytorch 0.4  
  
## My Solutions  
### Pre-processing & Augmentation  
1. Symmetric padding: 101\*101 \to 128\*128
2. Horizontal flip
3. Adjust brightness/contrast/gamma

### Test Time Augmentation  
1. null + horizontal flip

## Model Architecture  
  * Basic architecture: UNet  
  * Self-designed encoder/decoder: 
    * replaced every maxpool layer with an Inception module; 
    * replaced every conv block with a depthwise ResNeXt module;
    * no dropout
  * introduced a binary classification output head after encoder, as a deep supervision(thanks to Heng's idea);
  * scSE blocks were attached after each contracting/expanding block;

## Training
  * Model were trained with an ensembled loss: LovaszHingeLoss(images with non-empty masks) + BCELoss(binaryclassification) + LovaszHingeLoss(all images) with weights (0.1, 0.1, 1.0);
  * SGD with initial lr 0.1 as the optimizer;
  * CosineAnnealing lr to 1e-5 in each cycle, 80/120/160 epochs per cycle;
  * batch_size = 16;
  * 5 fold cross validation;
  * snapshot from each cycle + each fold for ensemble;

## Performance
  * Public Leaderboard: 265th/3234; score: 0.8424
  * Private Leaderboard: 230th/3234; score: 0.8660

## Things Might Improve the Performance
  * Hypercolumn. Tried it in earlier days but abandoned, it significantly slowed down the training;
  * Pseudo Labelling;
  * Deeper (pretrained) encoder;
  * Post-processing.
