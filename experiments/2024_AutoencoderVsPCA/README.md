A PyTorch implementation of the autoencoder architecture described in
> Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. Science, 313(5786), 504-507.

## Architecture
The network reduces 784-dimensional MNIST images to a 30-dimensional code space and then reconstructs the original images.
- **Encoder**: 784 → 1000 → 500 → 250 → 30
- **Decoder**: 30 → 250 → 500 → 1000 → 784

## Training
```bash
python hae.py --epochs 100 --batch-size 256 --device cuda --save-model --output-dir ./hae_results
```
## PCA Comparison
A script to compare the autoencoder's performance against PCA and Logistic PCA.

```bash
python hae_pca.py --n-components 30 --device cuda --batch-size 128 --output-dir ./hae_pca_results
```
