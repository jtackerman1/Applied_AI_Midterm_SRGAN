# Applied AI Midterm SRGAN Project

## Project objective

Build a reproducible PyTorch project that compares two binary image
classifiers:

- Model A: transfer-learning classifier trained on original images resized
  to 128x128.
- Model B: transfer-learning classifier trained on 128x128 images generated
  by a Super Resolution GAN from 32x32 inputs.

## Required experimental rules

1. Use PyTorch and torchvision.
2. Split the original dataset once into 70% training and 30% testing.
3. Use a stratified split with random seed 42.
4. Save the resulting file paths and labels so every notebook uses the same
   split.
5. Never use test images to train the classifiers or SRGAN.
6. Model A receives original images resized to 128x128.
7. SRGAN inputs are 32x32 and targets are 128x128.
8. Train SRGAN for at least 150 epochs.
9. Save resumable checkpoints every 5 epochs.
10. A checkpoint must contain:
    - epoch
    - generator state
    - discriminator state
    - both optimizer states
    - both scheduler states when present
    - training history
    - random seed/configuration
11. Model B is trained on generator-produced 128x128 training images.
12. Evaluate Models A and B using the same reserved test examples.
13. Report accuracy, precision, recall, F1, ROC AUC, confusion matrices,
    classification reports and ROC curves.
14. Visualize:
    - original images
    - normalized and augmented images
    - 32x32 low-resolution inputs
    - bicubic 128x128 images
    - generated 128x128 images
    - real 128x128 targets
15. Use device-independent code supporting CUDA, Apple MPS and CPU.
16. Use pathlib rather than hard-coded path separators.
17. Use type hints and concise docstrings.
18. Put reusable code under src/applied_ai_midterm.
19. Keep notebooks lightweight; notebooks should call reusable source modules.
20. Add clear error messages for missing data and invalid directory structures.

## Classifier requirements

Use transfer learning with a torchvision pretrained model such as
MobileNetV2 or ResNet18.

The final layer must produce one binary logit. Use BCEWithLogitsLoss and
apply sigmoid only when generating probabilities.

Use ImageNet normalization when using ImageNet pretrained weights.

## SRGAN requirements

Create separate Generator and Discriminator classes.

The generator should include:
- initial convolution
- residual blocks
- skip connection
- two 2x upsampling blocks for total 4x enlargement
- final RGB output
- tanh or otherwise clearly documented output range

The discriminator should include progressively deeper convolution blocks and
produce real/fake logits.

Use a clearly documented generator loss consisting of:
- pixel/content reconstruction loss
- adversarial loss
- optional perceptual loss using pretrained VGG features

Do not download pretrained weights inside unit tests.

## Repository safety

Do not commit:
- raw datasets
- generated image datasets
- model checkpoint binaries
- Colab credentials
- Kaggle credentials
- Google Drive credentials
- environment folders

Before modifying several files, explain the intended changes.

After each implementation phase:
1. Run relevant tests.
2. Run Ruff or another static check.
3. Show a summary of files changed.
4. Do not create a Git commit unless explicitly requested.
