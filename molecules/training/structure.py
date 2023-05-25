import torch
import numpy as np


class StructuredDataBatch():
    def __init__(self, tuple_batch, dims, observed, exist, is_onehot, graphical_structure):
        """
            Container for a batch of data alongside structural information

            tuple_batch:         tuple of tensors containing the batch of data. Batch dimension B. Length of tuple (K)
            dims:                tensor of length (B,) containing the dimension of each datapoint in the batch
                                    Note that this is not strictly the dimension of the flattened tensor but is a
                                    problem dependent dimension index that dictates which 'dimension class' the datapoint is
            observed:            numpy array (K,) of 0 or 1 for whether that tensor is observed 
            exist:               list (K,) whether a tensor 'exists'
            is_onehot            list (K,) of 0 or 1 for whether that tensor is onehot encoded
            shapes               list (K,) of shapes for each tensor in the datapoint
            graphical_structure  an object describing dataset specific graphical_structure
        """
        self.exist = np.array(exist, dtype=np.uint8)
        self.observed = np.array([o for o, e in zip(observed, self.exist) if e], dtype=np.uint8)
        self.latent = 1 - self.observed
        self.is_onehot = [oh for oh, e in zip(is_onehot, self.exist) if e]
        self.gs = graphical_structure

        self.tuple_batch = tuple_batch
        self._dims = dims

        # self.max_dim = self.dims.max()

        self.B = self.tuple_batch[0].shape[0]
        self.K = len(self.tuple_batch)
        assert self._dims.shape[0] == self.B
        assert len(self._dims.shape) == 1

    def create_repeated_copy(self, K):
        copy = self.create_copy(self)
        copy.tuple_batch = tuple(torch.cat([t for _ in range(K)], dim=0) for t in self.tuple_batch)
        copy._dims = torch.cat([self._dims for _ in range(K)], dim=0)
        return copy

    @classmethod
    def create_copy(cls, original):
        tuple_batch_copy = tuple(t.clone() for t in original.tuple_batch)
        return StructuredDataBatch(tuple_batch_copy, original._dims.clone(), original.observed, original.exist, original.is_onehot, original.gs)


    def get_tuple_batch(self):
        return self.tuple_batch
    
    def get_flat_batch(self, select):
        return torch.cat([t.flatten(start_dim=1) for t, s in zip(self.tuple_batch, select) if s], dim=1)

    def get_flat_lats(self):
        return self.get_flat_batch(1-self.observed)

    def get_tuple_obs(self):
        return tuple(t for t, o in zip(self.tuple_batch, self.observed) if o)

    def get_flat_lats_and_obs(self):
        return self.get_flat_lats(), self.get_tuple_obs()

    @property
    def latent_dim(self):
        # return sum(np.prod(shape) for shape, o in zip(self.gs.shapes_with_onehot(self.max_dim), self.observed) if not o)
        return sum(np.prod(shape) for shape, o in zip(self.gs.shapes_with_onehot(), self.observed) if not o)

    def to(self, device):
        # move tensors to device
        self.tuple_batch = tuple(t.to(device) for t in self.tuple_batch)

    def get_device(self):
        return self.tuple_batch[0].device

    def get_dims(self):
        return self._dims

    def set_dims(self, new_dims):
        self._dims = new_dims

    def delete_dims(self, new_dims):
        # deletes some dimensions of the data so that the output data has dimensions given by new_dims
        self.tuple_batch = self.gs.remove_problem_dims(self.tuple_batch, new_dims)
        self._dims = new_dims

        # self.max_dim = self.dims.max()
        # self.tuple_batch = self.gs.strip_padding(self.tuple_batch, self.max_dim)

    def set_flat_lats(self, new_flat_lats):
        data = []
        # for shape, l in zip(self.gs.shapes_with_onehot(self.max_dim), self.latent):
        for shape, l in zip(self.gs.shapes_with_onehot(), self.latent):
            if l:
                numel = np.prod(shape)
                t, new_flat_lats = new_flat_lats[:, :numel], new_flat_lats[:, numel:]
                data.append(t.reshape(-1, *shape))
            else:
                data.append(None)
        assert new_flat_lats.shape[1] == 0

        self.tuple_batch = tuple(tb if o else d for d, tb, o in zip(data, self.tuple_batch, self.observed))

    def add_dim_where_not_max(self):
        self.set_dims(self._dims + (self._dims < self.gs.max_problem_dim))

    def add_one_dim(self):
        # Adds one more dimension to each item in the batch
        raise NotImplementedError
        self.max_dim = self.max_dim + 1
        self.dims = self.dims + 1
        self.tuple_batch = self.gs.add_extra_dim(self.tuple_batch)

    def expand_max_dim_by_1(self):
        raise NotImplementedError
        # expands the max dim but doesn't change the inherent dimension of the data
        self.max_dim = self.max_dim + 1
        self.tuple_batch = self.gs.add_extra_dim(self.tuple_batch)

    def delete_one_dim(self):
        # self.max_dim = self.max_dim - 1
        self._dims = self._dims - 1
        self.tuple_batch = self.gs.remove_problem_dims(self.tuple_batch, self._dims)
        # self.tuple_batch = self.gs.strip_padding(self.tuple_batch, self.max_dim)
    

    def get_mask(self, B, include_onehot_channels, include_obs):
        # gets a flat mask that is 1 for dimensions corresponding to a non zeroed out dimension
        # if include_onehot_channels then we assume the flattened length includes flattened onehot channels
        # if incdlue_obs then we assume the flattened length includes flattened obsevations

        device = self.get_device()

        data = []
        # for shape, e, oh, o in zip(self.gs.shapes_with_onehot(self.max_dim), self.exist, self.is_onehot, self.observed):
        for shape, e, oh, o in zip(self.gs.shapes_with_onehot(), self.exist, self.is_onehot, self.observed):

            if o and not include_obs:
                data.append(None)
                continue

            if e:
                if oh:
                    if include_onehot_channels:
                        data.append(torch.ones((B, *shape), device=device))
                    else:
                        data.append(torch.ones((B, *shape[1:]), device=device))
                else:
                    data.append(torch.ones((B, *shape), device=device))

        data = self.gs.remove_problem_dims(data, self._dims)

        if include_obs:
            return torch.cat([t.flatten(start_dim=1) for t in data], dim=1)
        else:
            return torch.cat([t.flatten(start_dim=1) for t, o in zip(data, self.observed) if not o], dim=1)

    def get_next_dim_deleted_mask(self, B, include_onehot_channels, include_obs):
        """
            Gets a flat mask that is 1s for dimensions that would be deleted or set to 0 if we were to move down
            one dimension class 
        """
        outer_mask = self.get_mask(B, include_onehot_channels, include_obs)
        self._dims = self._dims - 1
        inner_mask = self.get_mask(B, include_onehot_channels, include_obs)
        self._dims = self._dims + 1
        return outer_mask - inner_mask

    def get_next_dim_added_mask(self, B, include_onehot_channels, include_obs):
        """
            Gets a flat mask that is 1s for dimensions that would be added if we were to move up
            one dimension class
        """
        inner_mask = self.get_mask(B, include_onehot_channels, include_obs)
        self._dims = self._dims + 1
        outer_mask = self.get_mask(B, include_onehot_channels, include_obs)
        self._dims = self._dims - 1
        return outer_mask - inner_mask

    def convert_problem_dim_to_tensor_dim(self, problem_dim_data):
        """
            problem_dim_data (B, problem_dim)
            e.g. for matrix factorization problem_dim is say 4 for up to 4x4 matrices
            but tensor dim is around 112 for all the flattened matrices
            output (B, tensor_dim) with values for each problem dim repeated an 
            appropriate amount of times for each correspondance
        """
        # populate an empty st_batch of dim 1 with all B, 0 values
        # then flatten
        # populate an empty st_batch of dim 2 with all B, 1 values
        # then flatten
        # add to first but using the next dim deleted mask

        # create a new st_batch with all ones and is full dim
        # get next dim deleted mask and all these elements get set to B, 3
        # then get dim below that mask and all these elements get set to B, 2

        B = problem_dim_data.shape[0]
        
        tmp = StructuredDataBatch.create_copy(self)
        tmp.set_dims(tmp.gs.max_problem_dim * torch.ones_like(tmp._dims))

        problem_dim_data_counter = -1
        while True:
            outer_mask = tmp.get_mask(B, include_obs=False, include_onehot_channels=True)
            tmp._dims = tmp._dims - 1
            inner_mask = tmp.get_mask(B, include_obs=False, include_onehot_channels=True)
            mask = outer_mask - inner_mask
            flat_lats = tmp.get_flat_lats()
            flat_lats = (1-mask) * flat_lats + mask * problem_dim_data[:, problem_dim_data_counter].view(-1, 1)
            tmp.set_flat_lats(flat_lats)

            problem_dim_data_counter = problem_dim_data_counter - 1

            if tmp._dims[0] == 0:
                break

        return tmp.get_flat_lats()









class Structure():
    def __init__(self, exist, observed, dataset):  #, shapes=None, example=None, names=None):
        """
        Stores metadata about tensor shapes and observedness. One of shapes or example (without batch dimension)
        must be provided to extract the shapes from.
        """
        self.exist = np.array(exist, dtype=np.uint8)
        self.observed = np.array([o for o, e in zip(observed, self.exist) if e], dtype=np.uint8)
        self.latent = 1 - self.observed
        self.is_onehot = [oh for oh, e in zip(dataset.is_onehot, self.exist) if e]
        names = dataset.names if hasattr(dataset, "names") else [f"tensor_{i}" for i in range(len(self.shapes))]
        self.names = [n for n, e in zip(names, self.exist) if e]
        print("Created structure with observedness", self.observed)
        if hasattr(dataset, "graphical_structure"):
            self.graphical_structure = dataset.graphical_structure

    def set_varying_problem_dims(self, problem_dims, batch_max_problem_dim):
        self.graphical_structure.set_varying_problem_dims(problem_dims, batch_max_problem_dim)
        self.shapes = self.graphical_structure.shapes

    def add_one_to_problem_dims(self):
        self.graphical_structure.add_one_to_problem_dims()
        self.shapes = self.graphical_structure.shapes

    def sub_one_from_problem_dims(self):
        self.graphical_structure.sub_one_from_problem_dims()
        self.shapes = self.graphical_structure.shapes

    def get_exist_mask(self, include_obs, include_onehot_channels):

        B = len(self.graphical_structure.problem_dims)
        # existing_data = [torch.ones((B, *shape)) for shape, e in zip(self.shapes, self.exist) if e]

        existing_data = []
        for shape, e, oh in zip(self.shapes, self.exist, self.is_onehot):
            if e:
                if oh:
                    if include_onehot_channels:
                        existing_data.append(torch.ones((B, *shape)))
                    else:
                        existing_data.append(torch.ones((B, *shape[1:])))
                else:
                    existing_data.append(torch.ones((B, *shape)))

        existing_data = self.graphical_structure.remove_problem_dims(existing_data)

        # return self.flatten_latents(existing_data, contains_marg=False)
        if include_obs:
            return self.get_flattened(existing_data, [1] * len(existing_data))
        else:
            return self.flatten_latents(existing_data, contains_marg=False)


    def get_exist_mask_after_deleting_dim(self, include_obs, include_onehot_channels):
        problem_dims = self.graphical_structure.problem_dims
        reduced_dims = (self.graphical_structure.problem_dims - 1).clamp(min=1)
        self.set_varying_problem_dims(reduced_dims, self.graphical_structure.batch_max_problem_dim)
        exist_after_deleting_next_mask = self.get_exist_mask(include_obs, include_onehot_channels)
        self.set_varying_problem_dims(problem_dims, self.graphical_structure.batch_max_problem_dim)
        return exist_after_deleting_next_mask

    def get_next_dim_deleted_mask(self, include_obs, include_onehot_channels):
        return self.get_exist_mask(include_obs, include_onehot_channels) - self.get_exist_mask_after_deleting_dim(include_obs, include_onehot_channels)

    def get_next_dim_added_mask(self, include_obs, include_onehot_channels):
        """
            Gets the mask for new dimensions that get added when increaseing the problem dimension by 1 
            Assumes batch_max_problem_dim is big enough to hold the bigger mask
        """
        current_problem_dims = self.graphical_structure.problem_dims
        current_batch_max_problem_dim = self.graphical_structure.batch_max_problem_dim

        self.set_varying_problem_dims(current_problem_dims+1, current_batch_max_problem_dim)
        big_mask = self.get_exist_mask(include_obs, include_onehot_channels)
        self.set_varying_problem_dims(current_problem_dims, current_batch_max_problem_dim)
        small_mask = self.get_exist_mask(include_obs, include_onehot_channels)

        return big_mask - small_mask

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
        arg = [a for a, e in zip(arg, structure.exist) if e]  # filter to only existent tensors
        self.tensorwise_arg = arg
        self.structure = structure
        self.dtype = dtype

    @property
    def lats(self):
        arg = tuple([a*torch.ones((1, *shape), dtype=self.dtype) for a, shape in zip(self.tensorwise_arg, self.structure.shapes)])
        return self.structure.flatten_latents(arg, contains_marg=False)

    @property
    def obs(self):
        arg = tuple([a*torch.ones((1, *shape), dtype=self.dtype) for a, shape in zip(self.tensorwise_arg, self.structure.shapes)])
        _, obs = self.structure.flatten_batch(arg, contains_marg=False)
        return obs