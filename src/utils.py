import matplotlib.pyplot as plt
import numpy as np


def generate_tints(clusters):
    num_clusters = 20
    colors = plt.cm.get_cmap("tab20", num_clusters)
    unique_clusters = np.unique(clusters)
    unique_clusters.sort()

    if len(unique_clusters) > num_clusters:
        raise ValueError(
            f"Number of unique clusters ({len(unique_clusters)}) exceeds available colors ({num_clusters})."
        )

    tint_map = {i: colors(i)[:3] for i in unique_clusters}
    return tint_map


def apply_tint(image_array, tint):
    image_array = np.ones((image_array.shape), dtype=np.float32)
    return np.clip(image_array * tint, 0, 1)


def create_image(
    cluster_ids,
    maxx,
    maxy,
    locx,
    locy,
    images,
    patch_size,
    subpatch_size,
    subpatches_in_patch,
):
    num_patches = len(locx)
    final_image = np.ones((maxy, maxx, 3), dtype=np.float32)
    tint_map = generate_tints(cluster_ids)

    for i in range(num_patches):
        patchx = locx[i].item()
        patchy = locy[i].item()

        for j in range(subpatches_in_patch):
            cluster_id = cluster_ids[i * subpatches_in_patch + j].item()
            tint = tint_map[cluster_id]
            subpatch = apply_tint(images[i, j].numpy() / 255.0, tint)

            sx = (patchx * patch_size) + (
                j % (patch_size // subpatch_size)
            ) * subpatch_size
            sy = (patchy * patch_size) + (
                j // (patch_size // subpatch_size)
            ) * subpatch_size

            final_image[sy : sy + subpatch_size, sx : sx + subpatch_size] = subpatch

    return final_image, tint_map
