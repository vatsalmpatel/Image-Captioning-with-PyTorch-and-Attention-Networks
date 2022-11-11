# Image 📸 Captioning with PyTorch and Attention Networks

Image Captioning is a big task that neural networks have gotten so good at in recent times and this goodness if you want to call it that is due to `transformer architecture` (encoders and decoders) used to solve the these problems. Image Captioning presented here is also based on transformer architecture from the [**Show, Attend and Tell**](https://arxiv.org/abs/1502.03044) Paper. This paper proposes almost the same architecture as its big brother [**Show and Tell**](https://arxiv.org/abs/1411.4555) but with a little change. It uses `Attention` on Images to caption specific portions of the Image instead of looking and learning from the entire Image. 

Attention can be of two types: `soft attention` and `hard attention`. In `Soft Attention`, probability is given to each part of the image to give it importance to learn something during training. In `Hard Attention`, each image is split into `different parts` and either that part is used during training or is not used at the time of training.

It uses a `pretrained CNN model` (encoder part) to learn features from the image and then pass it to a `RNN layer` (decoder part) with attention which is used to generate sequence of words (sentence) to caption the image. It also uses `Beam Search` in which you don't let your Decoder be lazy and simply choose the words with the best score at each decode-step. Beam Search is useful for any language modeling problem because it finds the most optimal sequence.

The model is trained on the [**Flickr8K**](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset, which containes nearly 8000 images and 5 captions per image. We use the data splits by `Andrej Karpathy`, which can be found [**here**](https://www.kaggle.com/datasets/adityajn105/flickr8k). The splits are for both the Flickr and the COCO dataset. So, if you have compute power and space, you can try the COCO dataset, which is way bigger than the Flickr Dataset.

The model is trained for `10` epochs with `learning rate deacy`. It needs to be trained for atleast 100 epochs in order for us to see better results and achieve a better `BLEU score`. The model trained for 10 epochs achieves a validation accuracy of around 60% and a BLEU score of 0.1, which means the model is to be trained for a hell lot more to achieve a better result, both in terms of image captions generated as well as a better score on the test dataset.

The high level picture of the entire model can look something like this (image taken from the paper):

![model_image_high_level](./simplified_model.jpg)