# Generative Models

This project explores two popular generative models: Variational Autoencoder (VAE) and Generative Adversarial Network (GAN). Both models are applied to the MNIST dataset, which consists of handwritten digits. The goal is to learn the underlying data distribution and generate new, realistic digit images that resemble the originals. The VAE focuses on learning a compressed latent representation with probabilistic elements, while the GAN uses a competitive training process between a generator and a discriminator to produce high-quality samples.

The project demonstrates the core principles of these models, including their architectures and the mathematical ideas that drive them. Below, we first outline the high-level workflow for each model, then delve into their architectures and the simple math behind how they function.

## Project Overview in Steps

1. **Data Preparation**: Load the MNIST dataset, which includes grayscale images of digits from 0 to 9. Normalize the pixel values to a range between 0 and 1, and reshape them into a format suitable for convolutional neural networks (e.g., adding a channel dimension for single-channel images).

2. **Model Building**: Construct the VAE and GAN models separately. For the VAE, build an encoder to map input images to a latent space and a decoder to reconstruct images from that space. For the GAN, create a generator to produce images from random noise and a discriminator to classify images as real or fake.

3. **Loss and Optimization Setup**: Define custom loss functions. The VAE uses a combination of reconstruction error and a regularization term to encourage a structured latent space. The GAN employs adversarial losses where the discriminator aims to correctly identify real and fake images, while the generator tries to fool the discriminator.

4. **Training Process**: Train the models iteratively. In the VAE, the entire network learns to minimize the combined loss. In the GAN, alternate between updating the discriminator on real and generated samples and updating the generator to improve its outputs based on the discriminator's feedback.

5. **Generation of New Samples**: After training, use the decoder in the VAE or the generator in the GAN to create new digit images by sampling from the latent space or random noise.

6. **Evaluation Considerations**: While the focus is on generation, qualitative assessment involves inspecting the visual quality of produced images, and quantitative metrics could include measures of diversity and fidelity to the original data distribution.

## Variational Autoencoder (VAE) Model

### High-Level Working
The VAE works by encoding input images into a probabilistic latent space and then decoding samples from that space to reconstruct or generate images. Unlike a standard autoencoder, it introduces randomness in the latent space to enable smooth interpolation and generation of new samples. The model balances reconstructing the input accurately while keeping the latent distribution close to a standard normal distribution.

### Model Architecture

#### Encoder
- Input: Grayscale image (28x28x1).
- Convolutional Layer 1: 32 filters, 3x3 kernel, stride 2, ReLU activation, same padding → Output: 14x14x32.
- Convolutional Layer 2: 64 filters, 3x3 kernel, stride 2, ReLU activation, same padding → Output: 7x7x64.
- Flatten: → Output: 3136-dimensional vector.
- Dense Layer: 128 units, ReLU activation → Output: 128-dimensional vector.
- Mean Output: Dense layer to latent dimension (e.g., 16) → Represents the mean of the latent distribution.
- Log Variance Output: Dense layer to latent dimension (e.g., 16) → Represents the log variance of the latent distribution.
- Sampling Layer: Samples from the normal distribution using the mean and log variance → Output: Latent vector (e.g., 16-dimensional).

#### Decoder
- Input: Latent vector (e.g., 16-dimensional).
- Dense Layer: Projects to 7x7x64 → Output: 3136-dimensional vector.
- Reshape: → Output: 7x7x64.
- Transpose Convolutional Layer 1: 64 filters, 3x3 kernel, stride 2, ReLU activation, same padding → Output: 14x14x64.
- Transpose Convolutional Layer 2: 32 filters, 3x3 kernel, stride 2, ReLU activation, same padding → Output: 28x28x32.
- Transpose Convolutional Layer 3: 1 filter, 3x3 kernel, stride 1, sigmoid activation, same padding → Output: Reconstructed image (28x28x1).

#### Overall VAE
- Combines encoder and decoder, with a custom loss layer applied to the outputs.

### Math Behind VAE
The VAE minimizes a loss that has two parts:

1. **Reconstruction Loss**: Measures how well the decoded image matches the original. It's the average binary cross-entropy between the input pixels and the reconstructed pixels. Simply put: Loss_recon = average( -x * log(y) - (1-x) * log(1-y) ) over all pixels, where x is the original pixel value (0 or 1) and y is the reconstructed value (between 0 and 1).

2. **KL Divergence (Regularization)**: Encourages the latent distribution to be close to a standard normal distribution (mean 0, variance 1). It's calculated as: KL = -0.5 * average(1 + log_var - mean^2 - exp(log_var)) over the latent dimensions. This term prevents the latent space from overfitting and promotes continuity.

Total Loss = Reconstruction Loss + KL Divergence.

During generation, sample a latent vector z from a normal distribution, then pass it through the decoder to get a new image.

## Generative Adversarial Network (GAN) Model

### High-Level Working
The GAN consists of two networks in competition: the generator creates fake images from random noise, and the discriminator evaluates whether images are real (from the dataset) or fake (from the generator). The discriminator gets better at detection, forcing the generator to improve its fakes. Over time, the generator learns to produce images indistinguishable from real ones.

### Model Architecture

#### Discriminator
- Input: Grayscale image (28x28x1).
- Convolutional Layer 1: 64 filters, 3x3 kernel, stride 2, LeakyReLU activation (alpha=0.2), same padding, followed by dropout (0.4) → Output: 14x14x64.
- Convolutional Layer 2: 64 filters, 3x3 kernel, stride 2, LeakyReLU activation (alpha=0.2), same padding, followed by dropout (0.4) → Output: 7x7x64.
- Flatten: → Output: 3136-dimensional vector.
- Dense Layer: 1 unit, sigmoid activation → Output: Probability (0 for fake, 1 for real).

#### Generator
- Input: Latent noise vector (e.g., 100-dimensional).
- Dense Layer: Projects to 7x7x128, LeakyReLU activation (alpha=0.2) → Output: 6272-dimensional vector.
- Reshape: → Output: 7x7x128.
- Transpose Convolutional Layer 1: 128 filters, 4x4 kernel, stride 2, LeakyReLU activation (alpha=0.2), same padding → Output: 14x14x128.
- Transpose Convolutional Layer 2: 128 filters, 4x4 kernel, stride 2, LeakyReLU activation (alpha=0.2), same padding → Output: 28x28x128.
- Convolutional Layer: 1 filter, 7x7 kernel, sigmoid activation, same padding → Output: Generated image (28x28x1).

#### Overall GAN
- Stacks the generator and discriminator (with discriminator frozen during generator updates).

### Math Behind GAN
The GAN optimizes two opposing losses using binary cross-entropy.

1. **Discriminator Loss**: Aims to classify real images as 1 and fake images as 0. For real samples: Loss_real = average( -log(D(real)) ), where D(real) is the discriminator's output for real images. For fake samples: Loss_fake = average( -log(1 - D(fake)) ). Total Discriminator Loss = Loss_real + Loss_fake.

2. **Generator Loss**: Aims to make the discriminator classify fake images as 1. Generator Loss = average( -log(D(fake)) ), where fake images are produced by the generator.

The training alternates: Update the discriminator to maximize its accuracy, then update the generator to minimize its loss against the updated discriminator.

During generation, input random noise to the generator to produce new images.

## Dependencies
- Python 3.x
- TensorFlow/Keras for model implementation
- NumPy for data handling
- Matplotlib for visualization (optional)
