import torch
import numpy as np
import copy



class Structure():
    def __init__(self, exist, observed, dataset):
        """
        Stores metadata about tensor shapes and observedness. One of shapes or example (without batch dimension)
        must be provided to extract the shapes from.
        """
        self.exist = np.array(exist, dtype=np.uint8)
        self.observed = np.array([o for o, e in zip(observed, self.exist) if e], dtype=np.uint8)
        self.latent = 1 - self.observed
        example = dataset.__getitem__(0, will_augment=False, do_duplicate=False)
        assert len(example) == len(self.exist)
        # if embedder is not None:
        #     example = [t.squeeze(0) for t in embedder([t.unsqueeze(0) for t in example])]
        #     n_added = len(example) - len(self.exist)
        #     emb_exist = np.ones(n_added, dtype=np.uint8) if emb_exist is None else np.array(emb_exist, dtype=np.uint8)
        #     assert len(emb_exist) == n_added
        #     self.exist = np.concatenate([self.exist, emb_exist])
        #     emb_observed = np.zeros(n_added, dtype=np.uint8) if emb_observed is None else np.array(emb_observed, dtype=np.uint8)
        #     assert len(emb_observed) == sum(emb_exist)
        #     self.observed = np.concatenate([self.observed, emb_observed])
        #     self.latent = 1 - self.observed
        self.shapes = [t.shape for t, e in zip(example, self.exist) if e]
        self.is_onehot = [oh for oh, e in zip(dataset.is_onehot, self.exist) if e]
        names = dataset.names if hasattr(dataset, "names") else []
        names += [f"tensor_{i}" for i in range(len(names), len(self.shapes))]
        self.names = [n for n, e in zip(names, self.exist) if e]
        print("Created structure with shapes", self.shapes, "and observedness", self.observed)
        # take graphical structure for GSDM
        if hasattr(dataset, "get_graphical_model_mask_and_indices"):
            self.graphical_model_mask_and_indices = dataset.get_graphical_model_mask_and_indices()
        if hasattr(dataset, "get_shareable_embedding_indices"):
            self.shareable_embedding_indices = dataset.get_shareable_embedding_indices()
        if hasattr(dataset, "dim_handler_class"):
            self.dim_handler = dataset.dim_handler_class(self.shapes)
        self.example = example

    def get_embedded_version(self, embed_func, new_exist=None, new_observed=None):
        example = self.example  # [torch.zeros(s) for s in self.shapes]
        embedded = [t.squeeze(0) for t in embed_func([t.unsqueeze(0) for t in example])]
        n_added = len(embedded) - len(self.exist)
        new_exist = np.ones(n_added, dtype=np.uint8) if new_exist is None else np.array(new_exist, dtype=np.uint8)
        assert len(new_exist) == n_added
        new_observed = np.zeros(n_added, dtype=np.uint8) if new_observed is None else np.array(new_observed, dtype=np.uint8)
        assert len(new_observed) == sum(new_exist)
        # add attributes to copy
        new_structure = copy.deepcopy(self)
        new_structure.exist = np.concatenate([self.exist, new_exist])
        new_structure.observed = np.concatenate([self.observed, new_observed])
        new_structure.latent = 1 - new_structure.observed
        new_structure.shapes = [t.shape for t, e in zip(embedded, new_structure.exist) if e]
        basic_names = [f"tensor_{i}" for i in range(len(new_structure.exist))]
        new_structure.names = [n for n, e in zip(basic_names, new_structure.exist) if e]
        return new_structure

    def filter(self, keep):
        """
        keep should be vector of bools, same length as self.observed (i.e. should not contain entries for marginalised tensors)
        """
        assert len(keep) == sum(self.exist)
        orig_exist = self.exist.copy()
        for i, e in enumerate(self.exist):
            keep_index = sum(orig_exist[:i])
            self.exist[i] = e and keep[keep_index]
        self.observed = np.array([o for o, k in zip(self.observed, keep) if k], dtype=np.uint8)
        self.latent = 1 - self.observed
        self.shapes = [s for s, k in zip(self.shapes, keep) if k]
        self.is_onehot = [oh for oh, k in zip(self.is_onehot, keep) if k]
        self.names = [n for n, k in zip(self.names, keep) if k]

    def add_node(self, obs, shape, is_onehot, name):
        self.exist = np.concatenate([self.exist, np.array([1], dtype=np.uint8)])
        self.observed = np.concatenate([self.observed, np.array([obs], dtype=np.uint8)])
        self.latent = 1 - self.observed
        self.shapes.append(shape)
        self.is_onehot.append(is_onehot)
        self.names.append(name)

    def reset_obs(self, observed):
        assert len(observed) == sum(self.exist)
        self.observed = np.array(observed, dtype=np.uint8)
        self.latent = 1 - self.observed

    @property
    def latent_names(self):
        return [n for n, l in zip(self.names, self.latent) if l]

    @staticmethod
    def get_flattened(batch, select):
        return torch.cat([t.flatten(start_dim=1) for t, s in zip(batch, select) if s], dim=1)

    def flatten_batch(self, batch, contains_marg):
        if contains_marg:
            batch = [t for t, e in zip(batch, self.exist) if e]
        lats = self.get_flattened(batch, 1-self.observed)
        obs = tuple(t for t, o in zip(batch, self.observed) if o)
        return lats, obs

    def flatten_latents(self, batch, contains_marg):
        if contains_marg:
            batch = [t for t, e in zip(batch, self.exist) if e]
        return self.get_flattened(batch, 1-self.observed)

    def flatten_obs(self, batch, contains_marg):
        if contains_marg:
            batch = [t for t, e in zip(batch, self.exist) if e]
        return tuple(t for t, o in zip(batch, self.observed) if o)

    def unflatten_batch(self, lats, obs, pad_marg):
        data = []
        for shape, o in zip(self.shapes, self.observed):
            numel = np.prod(shape)
            if o:
                t, obs = obs[0], obs[1:]
            else:
                t, lats = lats[:, :numel], lats[:, numel:]
            data.append(t.reshape(-1, *shape))
        if pad_marg:
            data_iter = iter(data)
            data = [next(data_iter) if e else None for e in self.exist]
        return tuple(data)

    def unflatten_latents(self, lats):
        data = []
        for shape, l in zip(self.shapes, self.latent):
            if l:
                numel = np.prod(shape)
                t, lats = lats[:, :numel], lats[:, numel:]
                data.append(t.reshape(-1, *shape))
        return tuple(data)

    @property
    def latent_dim(self):
        return sum(np.prod(shape) for shape, o in zip(self.shapes, self.observed) if not o)


class StructuredArgument():
    def __init__(self, arg, structure, dtype=torch.float32): 
        # arg should be a list of scalars. If it is a single scalar, or list of length 1, it is broadcasted.
        if type(arg) in [int, float]:
            arg = (arg,)
        if len(arg) == 1:
            arg = arg * len(structure.exist)
        assert len(arg) == len(structure.exist)
        self.cleaned_input = arg
        arg = [a for a, e in zip(arg, structure.exist) if e]  # filter to only existent tensors
        self.arg = tuple([a*torch.ones((1, *shape), dtype=dtype) for a, shape in zip(arg, structure.shapes)])
        self.structure = structure

    @property
    def lats(self):
        return self.structure.flatten_latents(self.arg, contains_marg=False)

    @property
    def obs(self):
        _, obs = self.structure.flatten_batch(self.arg, contains_marg=False)
        return obs
