<p align="center">
  <img width="538" height="700" src="https://github.com/user-attachments/assets/46d33d1e-20f9-40f4-92ed-94b21a7ea942" alt="image"/>
</p>


# Clustering Subpatch Embeddings for WSIs
Guided clustering UNI model subpatch embeddings. With a human in the loop, you can define semantically meaningful cluster centers, based on the features extracted with a UNI model. The cluster centers are then used as a segmentation model for other slides. Feature extracting model can be swapped for something else in `src/model.py`. 
To run the project you should run `cluster_subpatch.ipynb`. It uses samples from data/med_examples, but you can add your own .tiff files.

Currently only .tiff files are supported, but other file extensions are WIP. 
