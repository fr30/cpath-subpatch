import os
import slideflow as sf
import torch


def extract_tiles(filename, data_path):
    filename_noext = os.path.splitext(filename)[0]
    cache_dir = os.path.join("cache", data_path, filename_noext)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    tiles_path = os.path.join(cache_dir, f"{filename_noext}.tfrecords")

    if not os.path.exists(tiles_path):
        filepath = os.path.join(data_path, filename)
        wsi = sf.WSI(filepath, tile_px=224, tile_um="20x")
        wsi.qc(
            # "both",  # TODO: Change this to 'blur'
            "blur",
            filter_threshold=0.999,
            # blur_mpp=4.0,
            # blur_radius=5,
            # blur_threshold=0.01,
        )
        wsi.extract_tiles(
            tfrecord_dir=cache_dir,
            whitespace_fraction=1.0,
            grayspace_fraction=1.0,
            # grayspace_threshold=0.01,
            normalizer="reinhard",
            shuffle=False,
        )

    return sf.TFRecord(tiles_path)


def calculate_cluster_centers(features, cluster_ids):
    center_ids = cluster_ids.unique()
    num_clusters = center_ids.size(0)
    feature_dim = features.shape[-1]
    cluster_centers = torch.zeros((num_clusters, feature_dim), device=features.device)
    features_flat = features.view(-1, feature_dim)

    for i in center_ids:
        cluster_mask = cluster_ids == i
        cluster_features = features_flat[cluster_mask]
        cluster_centers[i] = cluster_features.mean(dim=0)

    return cluster_centers, center_ids


def calculate_cluster_ids(features, cluster_centers, center_ids):
    features_flat = features.view(-1, features.shape[-1])
    distances = torch.cdist(features_flat, cluster_centers)
    cluster_ids = distances.argmin(dim=1)
    cluster_ids = center_ids[cluster_ids]
    return cluster_ids
