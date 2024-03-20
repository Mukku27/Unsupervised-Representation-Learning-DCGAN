## Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to learn unsupervised representations of images from the MNIST dataset.

### Dependencies

This code requires the following Python libraries:

* torch
* torchvision
* torch.optim
* torch.utils.data
* tensorboard

### Files

* `model.py`: Contains the definitions for the Discriminator and Generator networks.
* `train.ipynb`: Implements the training loop for the DCGAN.

### Project Structure

The `model.py` file defines the Discriminator and Generator architectures. The Discriminator network takes an image as input and outputs a probability of whether the image is real or fake (generated). The Generator network takes random noise as input and generates an image that resembles the real data distribution.

The `train.ipynb` script trains the DCGAN in an alternating fashion. First, the Discriminator is trained to distinguish between real and fake images. Then, the Generator is trained to fool the Discriminator by generating images that the Discriminator classifies as real. This adversarial training process encourages the Generator to learn meaningful representations of the data.

### Training Process

The training script performs the following steps:

1. **Load Data:** Loads the MNIST dataset of handwritten digits.
2. **Define Models:** Initializes the Discriminator and Generator networks with weights.
3. **Define Optimizers:** Sets up Adam optimizers for both the Discriminator and Generator.
4. **Training Loop:**
    - For each epoch:
        - For each batch of images:
            - Train the Discriminator:
                - Generate fake images with the Generator.
                - Calculate the Discriminator loss for real and fake images.
                - Update the Discriminator weights.
            - Train the Generator:
                - Calculate the Generator loss based on the Discriminator's output.
                - Update the Generator weights.
    - Log training progress and generated images using TensorBoard.

### Usage

1. Save the code in two separate files: `model.py` and `train.ipynb`.
2. Install the required libraries (`torch`, `torchvision`, etc.).
3. Run the `train.ipynb` script to start training the DCGAN.
4. TensorBoard will be used to visualize the training progress and generated images. You can launch TensorBoard using `tensorboard --logdir=logs` (assuming the script is running in the same directory as the `logs` folder).

This is a basic implementation of a DCGAN for unsupervised representation learning. You can experiment with different hyperparameters (learning rate, batch size, network architectures) to improve the quality of the generated images.
check out the research paper :https://arxiv.org/pdf/1511.06434.pdf
