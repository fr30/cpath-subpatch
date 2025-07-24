import os
import slideflow as sf
import timm
import torch

from slideflow.model.extractors._factory_torch import TorchFeatureExtractor


class TensorPair:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def to(self, type):
        self.first = self.first.to(type)
        self.second = self.second.to(type)
        return self


class SubpatchVisionTransformer(torch.nn.Module):
    def __init__(self, timm_kwargs):
        super().__init__()
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
        )
        # self.model = timm.create_model(
        #     "hf_hub:MahmoodLab/conch", pretrained=True, **timm_kwargs
        # )
        self.model.forward = self.model.forward_features

    def forward(self, x):
        x_f = self.model.forward_features(x)
        x_p = self.model.forward_head(x_f)
        return TensorPair(x_f, x_p)


class UNI2FeatureExtractor(TorchFeatureExtractor):
    tag = "uni2"  # Unique identifier for registration

    def __init__(
        self,
        batch_size=32,
        patch_size=224,
        subpatch_size=14,
        emb_dim=1536,
        device="cuda",
    ):
        super().__init__()
        timm_kwargs = {
            "img_size": patch_size,
            "patch_size": subpatch_size,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": emb_dim,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        self.model = SubpatchVisionTransformer(timm_kwargs)
        # self.model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        self.model = self.model.to(device)
        self.model.eval()
        self.transform = self.build_transform(img_size=224, resize=224)
        self.preprocess_kwargs = {"standardize": False}
        self.num_features = emb_dim
        self.patch_size = patch_size
        self.subpatch_size = subpatch_size
        self.batch_size = batch_size
        self.device = torch.device(device)

    def dump_config(self):
        return self._dump_config(class_name="UNI2FeatureExtractor")

    def get_subpatch_features(self, tfr):
        tfr_path_noext = os.path.splitext(tfr.path)[0]
        cache_dir = os.path.dirname(tfr_path_noext)
        file_name_noext = os.path.basename(tfr_path_noext)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = UNI2FeatureExtractor(device=device)

        # num_patches = len(tfr) + 1
        # subpatches_in_patch = (self.patch_size // self.subpatch_size) ** 2

        subpatch_path = os.path.join(cache_dir, f"{file_name_noext}_subpatches.pt")
        features_path = os.path.join(
            cache_dir, f"{file_name_noext}_subpatch_features.pt"
        )
        locx_path = os.path.join(cache_dir, f"{file_name_noext}_locx.pt")
        locy_path = os.path.join(cache_dir, f"{file_name_noext}_locy.pt")

        if all(
            os.path.exists(p)
            for p in [subpatch_path, features_path, locx_path, locy_path]
        ):
            subpatch_features = torch.load(features_path)
            images = torch.load(subpatch_path)
            locx = torch.load(locx_path)
            locy = torch.load(locy_path)
        else:
            subpatch_features, images, locx, locy = self.create_subpatch_features(tfr)
            torch.save(subpatch_features, features_path)
            torch.save(images, subpatch_path)
            torch.save(locx, locx_path)
            torch.save(locy, locy_path)

        maxx = (locx.max().item() + 1) * self.patch_size
        maxy = (locy.max().item() + 1) * self.patch_size

        return (
            subpatch_features,
            images,
            locx,
            locy,
            maxx,
            maxy,
        )

    def create_subpatch_features(self, tfr):
        num_patches = len(tfr) + 1
        subpatches_in_patch = (self.patch_size // self.subpatch_size) ** 2
        register_tokens = 9

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = UNI2FeatureExtractor(device=device)

        batch = []
        locx = torch.zeros(num_patches, dtype=torch.int32)
        locy = torch.zeros(num_patches, dtype=torch.int32)
        subpatch_features = torch.zeros(
            num_patches, subpatches_in_patch, self.num_features, dtype=torch.float32
        )
        images = torch.zeros(
            num_patches,
            subpatches_in_patch,
            self.subpatch_size,
            self.subpatch_size,
            3,
            dtype=torch.uint8,
        )

        for i, record in enumerate(tfr, start=1):
            image = sf.io.decode_image(bytes(record["image_raw"])).to(self.device)
            subpatches = self._split_patch(image)

            locx[i - 1] = record["loc_x"] // (2 * self.patch_size)
            locy[i - 1] = record["loc_y"] // (2 * self.patch_size)

            images[i - 1] = subpatches
            batch.append(image)

            if i % self.batch_size == 0:
                img_batch = torch.stack(batch).to(self.device)

                with torch.no_grad():
                    f = self(img_batch)

                # First 9 tokens are the register tokens, they don't correspond to any subpatch
                subpatch_features[i - self.batch_size : i] = f.first[
                    :, register_tokens:, :
                ]
                # patch_features[i - batch_size:i] = f.second
                batch = []

        return subpatch_features, images, locx, locy

    def _split_patch(self, img):
        subpatches = []

        for i in range(0, img.shape[0], self.subpatch_size):
            for j in range(0, img.shape[1], self.subpatch_size):
                subpatch = img[
                    i : i + self.subpatch_size, j : j + self.subpatch_size
                ].cpu()
                subpatches.append(subpatch)

        return torch.stack(subpatches)


class ClusterSegmentationModel:
    def __init__(self, cluster_centers, center_ids):
        self.cluster_centers = cluster_centers
        self.center_ids = center_ids
        self.feature_dim = cluster_centers.shape[1]

    def __call__(self, features):
        distances = torch.cdist(features, self.cluster_centers)
        cluster_ids = distances.argmin(dim=1)
        cluster_ids = self.center_ids[cluster_ids]
        return cluster_ids

    def save(self, path):
        torch.save(
            {"cluster_centers": self.cluster_centers, "center_ids": self.center_ids},
            path,
        )

    @classmethod
    def load(cls, path):
        data = torch.load(path)
        model = cls(
            cluster_centers=data["cluster_centers"],
            center_ids=data["center_ids"],
        )
        return model
