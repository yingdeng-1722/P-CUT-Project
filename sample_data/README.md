## 🧪 Reproducibility (Sample Data)

We provide representative sample data in the `sample_data/` folder to facilitate reproducibility. 

To test the inference pipeline with the provided samples, run:
```bash
python inference.py \
  --oct sample_data/Control001.jpg \
  --mu_t sample_data/Control001_mu_t.npy \
  --mask sample_data/Control001_mask.npy \
  --weights weights/netG_epoch_300.pth \
  --output sample_data/Control001_vhe_output.png
Please download the pre-trained generator weights (netG_epoch_300.pth) from our Releases page(https://github.com/yingdeng-1722/P-CUT-Project/releases/tag/v1.0-weights) and place it in the weights/ directory before running the inference script.
