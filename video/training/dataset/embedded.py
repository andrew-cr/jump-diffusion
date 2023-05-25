import os
import torch
import torch.nn.functional as F
import sklearn.cluster
from tqdm import tqdm
import pickle
from .base import StructuredDatasetBase
from .image import ImageFolderDataset, base_image_dataset_kwargs, image_gettable_kwargs
from multiprocessing import Pool, TimeoutError



class EmbeddedImageDataset(ImageFolderDataset, StructuredDatasetBase):
    which_to_normalize = None  # must be set by subclass

    def __init__(self, n_clusters, n_comp, dataset_cache_dir, train_embedder, device='cuda', **kwargs):
        super().__init__(use_labels=True, normalize=True, **kwargs)
        self.n_clusters = n_clusters if len(n_clusters) > 1 else n_clusters * len(self.is_onehot)
        self.n_comp = n_comp if len(n_comp) > 1 else n_comp * len(self.is_onehot)
        self.dataset_cache_dir = dataset_cache_dir
        os.makedirs(self.dataset_cache_dir, exist_ok=True)
        self.train_embedder = train_embedder
        self.device = device
        # extra property for saving embedder - can be edited by subclass methods to be different from train_embedder
        self.save_embedder = train_embedder
        ### synchronise arguments for clustering, PCA, and normalisation
        # don't normalise if clustering
        self.which_to_normalize = [int(wtn and nc == 0) for wtn, nc in zip(self.which_to_normalize, self.n_clusters)]
        # tensors are onehot if clustering
        self.is_onehot = [int(oh or nc > 0) for oh, nc in zip(self.is_onehot, self.n_clusters)]
        ### load network
        self._init_network()
        ### set up k-means clusters
        self._load_clusters()
        ### set up PCA
        self._load_pca()
        ### set up normalizations
        self._load_normalizations()
        ### set up cache
        self._cache = {}

    def _init_network(self):
        raise NotImplementedError

    def _run_network(self, images):
        raise NotImplementedError

    def _cached_run_network_and_process_output(self, images, idxs, do_cache=True, cache_device='cpu'):
        if not do_cache:
            return self._process_raw_output(self._run_network(images))
        idxs = [idx.item() for idx in idxs]
        not_in_cache = [idx not in self._cache for idx in idxs]
        if any(not_in_cache):
            # run on all
            out = self._process_raw_output(self._run_network(images))
            for i, idx in enumerate(idxs):
                self._cache[idx] = tuple(tensor[i].to(cache_device) for tensor in out)
        fetched = [self._cache[idx] for idx in idxs]
        return tuple(torch.stack(tensors).to(images.device) for tensors in zip(*fetched))

    def get_network_state_dict(self):
        if self.save_embedder:
            return self.embedder.state_dict()
        else:
            return {}

    def load_network_state_dict(self, state_dict):
        print("Loading network state dict")
        if self.save_embedder:
            self.embedder.load_state_dict(state_dict)

    def _process_raw_output(self, data):
        # apply clustering
        for i, kmeans in enumerate(self.kmeans):  
            if kmeans is None:
                continue
            features = data[i]
            clusterer = self.kmeans[i]
            cluster_labels = clusterer.predict(features)
            cshape = cluster_labels.shape
            onehot = torch.zeros((cshape[0], clusterer.n_clusters, *cshape[1:]), device=data[i].device)
            onehot.scatter_(1, cluster_labels.type(torch.int64).unsqueeze(1), 1)
            data[i] = onehot
        # apply PCA
        for i, pca in enumerate(self.pca):
            if pca is None:
                continue
            features = data[i]
            data[i] = pca.transform(features)
        # normalise
        data = [act if mean is None else torch.clamp((act-mean)/(2*std+1e-8), min=-3, max=3) for act, (mean, std) in zip(data, self.normalizations)]
        data = [data.to(torch.float32) for data in data]  # everything should be float32
        return tuple(data)

    def inverse_transform(self, data):
        """ Undo the effects of normalisation and PCA (can't help with clustering...)
        """
        data = list(data)
        if sum(self.which_to_normalize) > 0:
            raise NotImplementedError("Undoing normalisation is not implemented.")
        # undo PCA
        for i, pca in enumerate(self.pca):
            if pca is None:
                continue
            features = data[i]
            data[i] = pca.inverse_transform(features)
        return tuple(data)

    def __getitem__(self, idx, will_augment=True, deterministic=False):
        """
        WARNING: we don't get valid outputs until after calling .augment(...) on this.
        """
        image, _ = super().__getitem__(idx)
        if will_augment:
            return (image, torch.tensor(idx))  # embedding is done after augmentation
        else:
            # embed now
            image = image.unsqueeze(0).to(self.device)
            batched_data = self._process_raw_output(self._run_network(image))
            return tuple(t.squeeze(0) for t in batched_data)

    def get_images(self, data):
        images = data[0]
        return self._unnormalise_images(images)
    
    def augment(self, data, augment_pipe):  # can we change this?
        images, idxs = data
        if augment_pipe is None:
            augment_labels = None
        else:
            images, augment_labels = augment_pipe(images)
        data = self._cached_run_network_and_process_output(images, idxs, do_cache=False)
        #data = self._process_raw_output(self._run_network(images))
        return data, augment_labels

    def _load_normalizations(self):
        if sum(self.which_to_normalize) == 0:
            self.normalizations = [(None, None)] * len(self.which_to_normalize)
            return
        normalization_path = os.path.join(self.dataset_cache_dir, "norms.pkl")
        if not os.path.exists(normalization_path):
            norms = self._compute_normalizations()
            pickle.dump(norms, open(normalization_path, 'wb'))
        norm_stats = pickle.load(open(normalization_path, 'rb'))
        assert len(norm_stats) == len(self.which_to_normalize)
        self.normalizations = []
        for (m, s), to_norm in zip(norm_stats, self.which_to_normalize):
            if to_norm:
                self.normalizations.append((m.to(self.device), s.to(self.device)))
            else:
                self.normalizations.append((None, None))

    @torch.no_grad()
    def _compute_normalizations(self):
        """
        Compute channel-wise mean, std for all tensors returned by _run_network.
        """
        def _sum_act(act):
            for dim in range(2, len(act.shape)):
                act = act.mean(dim=dim, keepdim=True)  # mean over spatial dims
            return act.sum(dim=0, keepdim=True)
        sum_mean = [0.] * len(self.is_onehot)
        sum_sqr = [0.] * len(self.is_onehot)
        loader = torch.utils.data.DataLoader(self, batch_size=self.embedder_batch_size, shuffle=False, num_workers=4, drop_last=False)
        n_images = 0
        print('Computing normalizations for classifier')
        for images, in tqdm(loader):
            acts = self._run_network(images.to(self.device))
            for i, act in enumerate(acts):
                sum_mean[i] += _sum_act(act)
                sum_sqr[i] += _sum_act(act**2)
            n_images += len(images)
        mean = [m/n_images for m in sum_mean]
        std = [torch.sqrt((s/n_images) - m**2) for m, s in zip(mean, sum_sqr)]
        return [(m.cpu(), s.cpu()) for m, s in zip(mean, std)]

    def _get_embs(self, tensor_i):
        loader = torch.utils.data.DataLoader(self, batch_size=self.embedder_batch_size, shuffle=False, num_workers=4, drop_last=False)
        embeddings = []
        print('Computing embeddings for clustering tensor', tensor_i)
        for images, *_ in tqdm(loader):
            batch_emb = self._run_network(images.to(self.device))[tensor_i]
            embeddings.append(batch_emb)
        return torch.cat(embeddings, dim=0)

    def _load_pca(self, n_comp=None, tensor_i=None):
        """
        Utility for fitting models that use PCA instead of directly parameterising high-dimensional distributions.
        """
        if tensor_i is None:
            self.pca = [self._load_pca(tensor_i=i, n_comp=n) if n > 0 else None for i, n in enumerate(self.n_comp)]
            return
        pca_path = os.path.join(self.dataset_cache_dir, f"pca_{tensor_i}_{n_comp}.pkl")
        if not os.path.exists(pca_path):
            pca = self._fit_pca(tensor_i, n_comp)
            pickle.dump(pca, open(pca_path, 'wb'))
        return pickle.load(open(pca_path, 'rb'))

    @torch.no_grad()
    def _fit_pca(self, tensor_i, n_comp):
        embeddings = self._get_embs(tensor_i)
        pca = PCA(n_components=n_comp).fit(embeddings)
        return pca

    def _load_clusters(self, n_clusters=None, tensor_i=None):
        """
        Utility for fitting models that use K-means clustering instead of directly
        parameterising high-dimensional distributions.
        """
        if tensor_i is None:
            self.kmeans = [self._load_clusters(n, tensor_i=i) if n > 0 else None for i, n in enumerate(self.n_clusters)]
            return
        cluster_path = os.path.join(self.dataset_cache_dir, f"clusters_{tensor_i}_{n_clusters}.pkl")
        if not os.path.exists(cluster_path):
            clusters = self._fit_clusters(tensor_i, n_clusters)
            pickle.dump(clusters, open(cluster_path, 'wb'))
        print('Loading clusters from', cluster_path)
        return pickle.load(open(cluster_path, 'rb'))

    @torch.no_grad()
    def _fit_clusters(self, tensor_i, n_clusters):
        embeddings = self._get_embs(tensor_i)
        kmeans = Clusterer(n_clusters=n_clusters, random_state=0).fit(embeddings)
        return kmeans

    def vis_clusters(self, dirname, augment_pipe, device):
        if sum(self.is_onehot) == 0:
            print('No discrete variables to cluster')
            return
        print('Visualising clusters')
        n_images = 200
        disc_index = self.is_onehot.index(1)  # only cluster by first discrete value, and we'll assume it is a scalar
        loader = torch.utils.data.DataLoader(self, batch_size=n_images, shuffle=False, num_workers=4, drop_last=False)
        data = next(iter(loader))
        data = tuple(t.to(device) for t in data)
        data, _ = self.augment(data, augment_pipe)
        images = data[0]
        onehot = data[disc_index]
        n_values = onehot.shape[1]
        disc = torch.argmax(onehot, dim=1)
        for c in range(n_values):
            cluster_images = self._unnormalise_images(images[disc==c])
            B, H, W, C = cluster_images.shape
            cluster_images = cluster_images.transpose(1, 0, 2, 3).reshape(H, W*B, C)
            PIL.Image.fromarray(cluster_images).save(os.path.join(dirname, f'cluster_{c}.png'))


class Clusterer(sklearn.cluster.MiniBatchKMeans):
    """
    A wrapper around sklearn's MiniBatchKMeans to work with torch.tensors type and with spatial dimensions.
    """

    @staticmethod
    def collapse_spatial_dims(X):
        B, C, *spatial_dims = X.shape
        flattened = X.permute(0, *range(2, X.ndim), 1).flatten(end_dim=-2)
        return flattened, spatial_dims

    @staticmethod
    def restore_spatial_dims(flattened, spatial_dims):
        # ndim = len(spatial_dims) + 2
        return flattened.view(-1, *spatial_dims)  #, C).permute(0, ndim-1, *range(1, ndim-1))

    def fit(self, X):
        flat, spatial_dims = self.collapse_spatial_dims(X)
        flat = flat.cpu().numpy()
        return super().fit(flat)

    def predict(self, X):
        flat, spatial_dims = self.collapse_spatial_dims(X)
        flat = flat.cpu().numpy()
        pred = super().predict(flat)
        pred = torch.from_numpy(pred).to(X.device)
        return self.restore_spatial_dims(pred, spatial_dims)



class RobustPCA(sklearn.decomposition.PCA):
    """
    A wrapper to retry PCA fitting if it times out.
    """
    def fit(self, X, n_tries=20, timeout=1800):
        for i in range(n_tries):
            with Pool(processes=1) as pool:
                res = pool.apply_async(sklearn.decomposition.PCA.fit, (self, X,))
                try:
                    pca = res.get(timeout=timeout)
                    break
                except TimeoutError:
                    print('Timed out on iteration {i}')
                    pca = None
        if pca is None:
            raise TimeoutError("PCA fitting failed after 20 attempts.")
        self.__dict__.update(pca.__dict__)
        return self


class PCA(RobustPCA):
    """
    A wrapper around sklearn's PCA to work with torch.tensors type and with spatial dimensions.
    """

    @staticmethod
    def collapse_spatial_dims(X):
        B, C, *spatial_dims = X.shape
        flattened = X.permute(0, *range(2, X.ndim), 1).flatten(end_dim=-2)
        return flattened, spatial_dims

    @staticmethod
    def restore_spatial_dims(flattened, spatial_dims):
        # ndim = len(spatial_dims) + 2
        _, C = flattened.shape
        return flattened.view(-1, C, *spatial_dims)  #, C).permute(0, ndim-1, *range(1, ndim-1))

    def fit(self, X):
        flat, spatial_dims = self.collapse_spatial_dims(X)
        flat = flat.cpu().numpy()
        return super().fit(flat)

    def transform(self, X):
        flat, spatial_dims = self.collapse_spatial_dims(X)
        flat = flat.cpu().numpy()
        transformed = super().transform(flat)
        transformed = torch.from_numpy(transformed).to(X.device)
        return self.restore_spatial_dims(transformed, spatial_dims)

    def inverse_transform(self, X):
        flat, spatial_dims = self.collapse_spatial_dims(X)
        flat = flat.cpu().numpy()
        transformed = super().inverse_transform(flat)
        transformed = torch.from_numpy(transformed).to(X.device)
        return self.restore_spatial_dims(transformed, spatial_dims)


class CLIPDataset(EmbeddedImageDataset):
    embedder_batch_size = 100
    which_to_normalize = [0, 0]
    is_onehot = [0, 0]
    is_image = [1, 0]
    names = ['image', 'embedding']
    is_embedding = [0, 1]

    def _init_network(self):
        import clip
        self.embedder, embedder_preprocess = clip.load("ViT-B/32", device=self.device)
        self.embedder_norm = embedder_preprocess.transforms[-1]
        self.embedder = self.embedder.to(self.device)
        self.embedder.eval()
        if self.train_embedder:
            # if we're training the embedder, make sure it's not got all gradients disabled
            assert any(p.requires_grad for p in self.embedder.parameters())
        else:
            for p in self.embedder.parameters():
                p.requires_grad = False

    def _run_network(self, images):
        inp = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        # images currently normalised to [-1, 1]. Change to [0, 1] and then use normaliser
        inp = self.embedder_norm((inp+1)/2)
        emb = self.embedder.encode_image(inp)
        return [images, emb]


def get_embedded_and_labelled_dataset(EmbeddedClass):
    """
    Takes a subclass of EmbeddedImageDataset and returns a modification so that, if labels are present
    in the ImageFolderDataset inherited from, they are returned in the final position of the tuple.
    """
    assert issubclass(EmbeddedClass, ImageFolderDataset)

    class NewClass(EmbeddedClass):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.which_to_normalize = self.which_to_normalize + [0]
            self.is_onehot = self.is_onehot + [1]
            self.is_image = self.is_image + [0]
            self.names = self.names + ['label']
            self.is_embedding = self.is_embedding + [0]

        def __getitem__(self, idx, will_augment=True, deterministic=False):
            """
            WARNING: we don't get valid outputs until after calling .augment(...) on this.
            """
            image, label = ImageFolderDataset.__getitem__(self, idx)
            if will_augment:
                return (image, label)  # embedding is done after augmentation
            else:
                # embed now
                image = image.unsqueeze(0).to(self.device)
                batched_data = self._process_raw_output(self._run_network(image))
                return tuple(t.squeeze(0) for t in batched_data) + (label,)

        def augment(self, data, augment_pipe):
            images, labels = data
            if augment_pipe is None:
                augment_labels = None
            else:
                images, augment_labels = augment_pipe(images)
            data = self._process_raw_output(self._run_network(images)) + (labels,)
            return data, augment_labels

    NewClass.__name__ = f'Labelled{EmbeddedClass.__name__}'
    NewClass.__qualname__ = f'Labelled{EmbeddedClass.__qualname__}'
    return NewClass

LabelledCLIPDataset = get_embedded_and_labelled_dataset(CLIPDataset)

extra_EmbeddedImageDataset_kwargs = set([  # (name, type, default)
    ("dataset_cache_dir", 'str', None),
    ("train_embedder", 'bool', False),
    ("n_clusters", 'parse_int_list', '0'),
    ("n_comp", 'parse_int_list', '0'),
])
embedded_datasets_to_kwargs = {
    CLIPDataset: set.union(base_image_dataset_kwargs, extra_EmbeddedImageDataset_kwargs),
    LabelledCLIPDataset: set.union(base_image_dataset_kwargs, extra_EmbeddedImageDataset_kwargs),
}
embedded_kwargs_gettable_from_dataset = {
    CLIPDataset: image_gettable_kwargs,
    LabelledCLIPDataset: image_gettable_kwargs,
}
