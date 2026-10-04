# CIFAR-10 CNN: measuring what each regularisation technique contributes

A convolutional neural network for CIFAR-10 image classification, built in PyTorch, with a controlled comparison of data augmentation, batch normalisation and dropout against a plain baseline.

**Result: 82.4% test accuracy** with a 3-block CNN of about 620,000 parameters, trained for 25 epochs on a free Colab T4 GPU.

This is the second of three portfolio projects. The first, [numpy-mlp-mnist](https://github.com/warrior-here/numpy-mlp-mnist), implements an MLP and backpropagation by hand in NumPy. This one moves to PyTorch and convolutions.

## Summary of findings

| Run | Epochs | Best val acc | Final train acc | Final val acc | Train/val gap |
|---|---|---|---|---|---|
| Baseline | 15 | 75.7% | 97.3% | 74.4% | 22.9 pts |
| + Augmentation | 25 | 81.4% | 83.8% | 80.2% | 3.6 pts |
| + Batch norm | 25 | **83.2%** | 84.7% | 83.0% | 1.8 pts |
| + Dropout 0.25 | 25 | 82.7% | 81.8% | 82.7% | -0.9 pts |
| + Dropout 0.5 | 25 | 81.5% | 79.4% | 80.8% | -1.4 pts |

Each run adds one technique to the run above it.

- **The baseline overfits badly.** Validation loss reaches its minimum at epoch 4 and then rises, while training accuracy climbs to 97%.
- **Data augmentation gives the largest gain**: +5.7 points of validation accuracy, and the train/validation gap falls from 22.9 to 3.6 points.
- **Batch normalisation adds +1.8 points** and speeds up early training (about 61% validation accuracy after one epoch, against about 50% without it).
- **Dropout does not help here.** With the gap already under 2 points there is little overfitting left to remove. At p = 0.5 the model under-fits within 25 epochs.

The selected model is augmentation + batch norm, chosen on validation accuracy. It was then evaluated once on the test set.

![Validation accuracy and loss for all runs](assets/comparison.png)

## Test set results

Test accuracy: **82.4%** (8,240 of 10,000 images), against 83.0% on validation.

![Confusion matrix on the test set](assets/confusion_matrix.png)

| Class | Accuracy | Class | Accuracy |
|---|---|---|---|
| automobile | 95.9% | deer | 83.6% |
| frog | 92.1% | truck | 83.0% |
| ship | 88.7% | airplane | 80.9% |
| horse | 86.5% | bird | 77.5% |
| | | dog | 70.4% |
| | | cat | 65.4% |

- Cat and dog are the most confused pair: 144 dogs predicted as cats and 108 cats predicted as dogs.
- 95 trucks are predicted as automobiles, but only 20 automobiles as trucks.
- Most errors stay within a group: animals are confused with animals, and vehicles with vehicles.

## Method

**Data.** CIFAR-10: 60,000 colour images, 32x32 pixels, 10 classes. The 50,000 training images are split 45,000 / 5,000 into training and validation with a fixed seed. The 10,000 test images are held out until the end. Inputs are normalised per channel using statistics from the 45,000 training images only.

**Model.** Three convolution blocks (32, 64, 128 filters). Each block is a 3x3 convolution, optional batch norm, ReLU and 2x2 max-pooling. The classifier head is 2,048 -> 256 -> 10 with optional dropout after the hidden layer. The model has 620,586 parameters with batch norm enabled.

**Training.** Adam optimiser, learning rate 0.001, batch size 128, cross-entropy loss. Augmentation is a random crop (padding 4) and a random horizontal flip, applied to training images only.

**Protocol.** All choices (which techniques to keep, dropout rate, final model) were made on the validation set. The test set was evaluated once.

## Limitations

- **One seed per configuration.** With 5,000 validation images, sampling noise is about +/-0.5 points. The gains from augmentation and batch norm are well outside that; the difference between batch norm alone and dropout 0.25 is not.
- **Unequal epochs.** The baseline ran for 15 epochs and the other runs for 25. The baseline had peaked by epoch 10, and runs are compared on best validation accuracy.
- **No per-run tuning.** Learning rate and batch size were fixed at common defaults. Dropout was tried at two rates in one position, so the conclusion is limited to this setting.
- **Still improving at epoch 25.** The regularised runs had not plateaued, so longer training would likely raise accuracy.

## Possible next steps

- Repeat each configuration over several seeds and report mean and standard deviation.
- Train longer with a learning-rate schedule.
- Add residual connections and more depth.
- Add weight decay and label smoothing.

## Repository contents

| File | Purpose |
|---|---|
| `cifar10_cnn.ipynb` | Full notebook: data, model, training loop, all five runs, evaluation |
| `results.json` | Per-epoch loss and accuracy for every run |
| `cnn_cifar10.pt` | Trained weights of the selected model |
| `assets/` | Comparison plot and confusion matrix |
| `requirements.txt` | Python dependencies |

## How to run

1. Open `cifar10_cnn.ipynb` in Google Colab.
2. Select Runtime -> Change runtime type -> T4 GPU.
3. Run all cells. The five training runs take about 45 minutes in total.

To load the trained weights, define the `ConvBlock` and `CNN` classes from the notebook, then:

```python
model = CNN(use_bn=True)
model.load_state_dict(torch.load("cnn_cifar10.pt", map_location="cpu"))
model.eval()
```
