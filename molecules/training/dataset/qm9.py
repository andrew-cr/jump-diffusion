# This code is adapted from
# https://github.com/ehoogeboom/e3_diffusion_for_molecules


import torch
import numpy as np

from os.path import join as join
import os
import logging
import urllib.request
import tarfile
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
from math import inf
from .base import StructuredDatasetBase, GraphicalStructureBase
import wandb
from argparse import Namespace
from training.egnn_utils import DistributionNodes, random_rotation
import socket
import pickle


charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}




def get_unique_charges(charges):
    """
    Get count of each charge for each molecule.
    """
    # Create a dictionary of charges
    charge_counts = {z: np.zeros(len(charges), dtype=np.int)
                     for z in np.unique(charges)}
    print(charge_counts.keys())

    # Loop over molecules, for each molecule get the unique charges
    for idx, mol_charges in enumerate(charges):
        # For each molecule, get the unique charge and multiplicity
        for z, num_z in zip(*np.unique(mol_charges, return_counts=True)):
            # Store the multiplicity of each charge in charge_counts
            charge_counts[z][idx] = num_z

    return charge_counts

def add_thermo_targets(data, therm_energy_dict):
    """
    Adds a new molecular property, which is the thermochemical energy.

    Parameters
    ----------
    data : ?????
        QM9 dataset split.
    therm_energy : dict
        Dictionary of thermochemical energies for relevant properties found using :get_thermo_dict:
    """
    # Get the charge and number of charges
    charge_counts = get_unique_charges(data['charges'])

    # Now, loop over the targets with defined thermochemical energy
    for target, target_therm in therm_energy_dict.items():
        thermo = np.zeros(len(data[target]))

        # Loop over each charge, and multiplicity of the charge
        for z, num_z in charge_counts.items():
            if z == 0:
                continue
            # Now add the thermochemical energy per atomic charge * the number of atoms of that type
            thermo += target_therm[z] * num_z

        # Now add the thermochemical energy as a property
        data[target + '_thermo'] = thermo

    return data


def get_thermo_dict(gdb9dir, cleanup=True):
    """
    Get dictionary of thermochemical energy to subtract off from
    properties of molecules.

    Probably would be easier just to just precompute this and enter it explicitly.
    """
    # Download thermochemical energy
    logging.info('Downloading thermochemical energy.')
    gdb9_url_thermo = 'https://springernature.figshare.com/ndownloader/files/3195395'
    gdb9_txt_thermo = join(gdb9dir, 'atomref.txt')

    urllib.request.urlretrieve(gdb9_url_thermo, filename=gdb9_txt_thermo)

    # Loop over file of thermochemical energies
    therm_targets = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']

    # Dictionary that
    id2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    # Loop over file of thermochemical energies
    therm_energy = {target: {} for target in therm_targets}
    with open(gdb9_txt_thermo) as f:
        for line in f:
            # If line starts with an element, convert the rest to a list of energies.
            split = line.split()

            # Check charge corresponds to an atom
            if len(split) == 0 or split[0] not in id2charge.keys():
                continue

            # Loop over learning targets with defined thermochemical energy
            for therm_target, split_therm in zip(therm_targets, split[1:]):
                therm_energy[therm_target][id2charge[split[0]]
                                           ] = float(split_therm)

    # Cleanup file when finished.
    cleanup_file(gdb9_txt_thermo, cleanup)

    return therm_energy

def process_xyz_gdb9(datafile):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    mol_xyz = xyz_lines[2:num_atoms+2]
    mol_freq = xyz_lines[num_atoms+2]

    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
        atom_charges.append(charge_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])

    prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    prop_strings = prop_strings[1:]
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
    molecule.update(mol_props)
    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    return molecule

# Check if a string can be converted to an int, without throwing an error.
def is_int(str):
    try:
        int(str)
        return True
    except:
        return False

def process_xyz_files(data, process_file_fn, file_ext=None, file_idx_list=None, stack=True):
    """
    Take a set of datafiles and apply a predefined data processing script to each
    one. Data can be stored in a directory, tarfile, or zipfile. An optional
    file extension can be added.

    Parameters
    ----------
    data : str
        Complete path to datafiles. Files must be in a directory, tarball, or zip archive.
    process_file_fn : callable
        Function to process files. Can be defined externally.
        Must input a file, and output a dictionary of properties, each of which
        is a torch.tensor. Dictionary must contain at least three properties:
        {'num_elements', 'charges', 'positions'}
    file_ext : str, optional
        Optionally add a file extension if multiple types of files exist.
    file_idx_list : ?????, optional
        Optionally add a file filter to check a file index is in a
        predefined list, for example, when constructing a train/valid/test split.
    stack : bool, optional
        ?????
    """
    logging.info('Processing data file: {}'.format(data))
    if tarfile.is_tarfile(data):
        tardata = tarfile.open(data, 'r')
        files = tardata.getmembers()

        readfile = lambda data_pt: tardata.extractfile(data_pt)

    elif os.is_dir(data):
        files = os.listdir(data)
        files = [os.path.join(data, file) for file in files]

        readfile = lambda data_pt: open(data_pt, 'r')

    else:
        raise ValueError('Can only read from directory or tarball archive!')

    # Use only files that end with specified extension.
    if file_ext is not None:
        files = [file for file in files if file.endswith(file_ext)]

    # Use only files that match desired filter.
    if file_idx_list is not None:
        files = [file for idx, file in enumerate(files) if idx in file_idx_list]

    # Now loop over files using readfile function defined above
    # Process each file accordingly using process_file_fn

    molecules = []

    for file in files:
        with readfile(file) as openfile:
            molecules.append(process_file_fn(openfile))

    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    # Convert list-of-dicts to dict-of-lists
    molecules = {prop: [mol[prop] for mol in molecules] for prop in props}

    # If stacking is desireable, pad and then stack.
    if stack:
        molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}

    return molecules


# Cleanup. Use try-except to avoid race condition.
def cleanup_file(file, cleanup=True):
    if cleanup:
        try:
            os.remove(file)
        except OSError:
            pass



def gen_splits_gdb9(gdb9dir, cleanup=True):
    """
    Generate GDB9 training/validation/test splits used.

    First, use the file 'uncharacterized.txt' in the GDB9 figshare to find a
    list of excluded molecules.

    Second, create a list of molecule ids, and remove the excluded molecule
    indices.

    Third, assign 100k molecules to the training set, 10% to the test set,
    and the remaining to the validation set.

    Finally, generate torch.tensors which give the molecule ids for each
    set.
    """
    logging.info('Splits were not specified! Automatically generating.')
    gdb9_url_excluded = 'https://springernature.figshare.com/ndownloader/files/3195404'
    gdb9_txt_excluded = join(gdb9dir, 'uncharacterized.txt')
    urllib.request.urlretrieve(gdb9_url_excluded, filename=gdb9_txt_excluded)

    # First get list of excluded indices
    excluded_strings = []
    with open(gdb9_txt_excluded) as f:
        lines = f.readlines()
        excluded_strings = [line.split()[0]
                            for line in lines if len(line.split()) > 0]

    excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

    assert len(excluded_idxs) == 3054, 'There should be exactly 3054 excluded atoms. Found {}'.format(
        len(excluded_idxs))

    # Now, create a list of indices
    Ngdb9 = 133885
    Nexcluded = 3054

    included_idxs = np.array(
        sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

    # Now generate random permutations to assign molecules to training/validation/test sets.
    Nmols = Ngdb9 - Nexcluded

    Ntrain = 100000
    Ntest = int(0.1*Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    # Generate random permutation
    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)

    # Now use the permutations to generate the indices of the dataset splits.
    # train, valid, test, extra = np.split(included_idxs[data_perm], [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    train, valid, test, extra = np.split(
        data_perm, [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    assert(len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
        len(train), len(valid), len(test), len(extra))

    train = included_idxs[train]
    valid = included_idxs[valid]
    test = included_idxs[test]

    splits = {'train': train, 'valid': valid, 'test': test}

    # Cleanup
    cleanup_file(gdb9_txt_excluded, cleanup)

    return splits

def download_dataset_qm9(datadir, dataname, splits=None, calculate_thermo=True, exclude=True, cleanup=True):
    """
    Download and prepare the QM9 (GDB9) dataset.
    """
    # Define directory for which data will be output.
    gdb9dir = join(*[datadir, dataname])

    # Important to avoid a race condition
    os.makedirs(gdb9dir, exist_ok=True)

    logging.info(
        'Downloading and processing GDB9 dataset. Output will be in directory: {}.'.format(gdb9dir))

    logging.info('Beginning download of GDB9 dataset!')
    gdb9_url_data = 'https://springernature.figshare.com/ndownloader/files/3195389'
    gdb9_tar_data = join(gdb9dir, 'dsgdb9nsd.xyz.tar.bz2')
    # gdb9_tar_file = join(gdb9dir, 'dsgdb9nsd.xyz.tar.bz2')
    # gdb9_tar_data =
    # tardata = tarfile.open(gdb9_tar_file, 'r')
    # files = tardata.getmembers()
    urllib.request.urlretrieve(gdb9_url_data, filename=gdb9_tar_data)
    logging.info('GDB9 dataset downloaded successfully!')

    # If splits are not specified, automatically generate them.
    if splits is None:
        splits = gen_splits_gdb9(gdb9dir, cleanup)

    # Process GDB9 dataset, and return dictionary of splits
    gdb9_data = {}
    for split, split_idx in splits.items():
        gdb9_data[split] = process_xyz_files(
            gdb9_tar_data, process_xyz_gdb9, file_idx_list=split_idx, stack=True)

    # Subtract thermochemical energy if desired.
    if calculate_thermo:
        # Download thermochemical energy from GDB9 dataset, and then process it into a dictionary
        therm_energy = get_thermo_dict(gdb9dir, cleanup)

        # For each of train/validation/test split, add the thermochemical energy
        for split_idx, split_data in gdb9_data.items():
            gdb9_data[split_idx] = add_thermo_targets(split_data, therm_energy)

    # Save processed GDB9 data into train/validation/test splits
    logging.info('Saving processed data:')
    for split, data in gdb9_data.items():
        savedir = join(gdb9dir, split+'.npz')
        np.savez_compressed(savedir, **data)

    logging.info('Processing/saving complete!')


def prepare_dataset(datadir, dataset, subset=None, splits=None, cleanup=True, force_download=False):
    """
    Download and process dataset.

    Parameters
    ----------
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : dict, optional
        Dataset splits to use.
    cleanup : bool, optional
        Clean up files created while preparing the data.
    force_download : bool, optional
        If true, forces a fresh download of the dataset.

    Returns
    -------
    datafiles : dict of strings
        Dictionary of strings pointing to the files containing the data. 

    Notes
    -----
    TODO: Delete the splits argument?
    """

    # If datasets have subsets,
    if subset:
        dataset_dir = [datadir, dataset, subset]
    else:
        dataset_dir = [datadir, dataset]

    # Names of splits, based upon keys if split dictionary exists, elsewise default to train/valid/test.
    split_names = splits.keys() if splits is not None else [
        'train', 'valid', 'test']

    # Assume one data file for each split
    datafiles = {split: os.path.join(
        *(dataset_dir + [split + '.npz'])) for split in split_names}

    # Check datafiles exist
    datafiles_checks = [os.path.exists(datafile)
                        for datafile in datafiles.values()]

    # Check if prepared dataset exists, and if not set flag to download below.
    # Probably should add more consistency checks, such as number of datapoints, etc...
    new_download = False
    if all(datafiles_checks):
        logging.info('Dataset exists and is processed.')
    elif all([not x for x in datafiles_checks]):
        # If checks are failed.
        new_download = True
    else:
        raise ValueError(
            'Dataset only partially processed. Try deleting {} and running again to download/process.'.format(os.path.join(dataset_dir)))

    # If need to download dataset, pass to appropriate downloader
    if new_download or force_download:
        logging.info('Dataset does not exist. Downloading!')
        if dataset.lower().startswith('qm9'):
            download_dataset_qm9(datadir, dataset, splits, cleanup=cleanup)
        elif dataset.lower().startswith('md17'):
            raise NotImplementedError
            # download_dataset_md17(datadir, dataset, subset,
            #                       splits, cleanup=cleanup)
        else:
            raise ValueError(
                'Incorrect choice of dataset! Must chose qm9/md17!')

    return datafiles



def _get_species(datasets, ignore_check=False):
    """
    Generate a list of all species.

    Includes a check that each split contains examples of every species in the
    entire dataset.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.  Each dataset is a dict of arrays containing molecular properties.
    ignore_check : bool
        Ignores/overrides checks to make sure every split includes every species included in the entire dataset

    Returns
    -------
    all_species : Pytorch tensor
        List of all species present in the data.  Species labels shoguld be integers.

    """
    # Get a list of all species in the dataset across all splits
    all_species = torch.cat([dataset['charges'].unique()
                             for dataset in datasets.values()]).unique(sorted=True)

    # Find the unique list of species in each dataset.
    split_species = {split: species['charges'].unique(
        sorted=True) for split, species in datasets.items()}

    # If zero charges (padded, non-existent atoms) are included, remove them
    if all_species[0] == 0:
        all_species = all_species[1:]

    # Remove zeros if zero-padded charges exst for each split
    split_species = {split: species[1:] if species[0] ==
                     0 else species for split, species in split_species.items()}

    # Now check that each split has at least one example of every atomic spcies from the entire dataset.
    if not all([split.tolist() == all_species.tolist() for split in split_species.values()]):
        # Allows one to override this check if they really want to. Not recommended as the answers become non-sensical.
        if ignore_check:
            logging.error(
                'The number of species is not the same in all datasets!')
        else:
            raise ValueError(
                'Not all datasets have the same number of species!')

    # Finally, return a list of all species
    return all_species


class ProcessedDataset(Dataset):
    """
    Data structure for a pre-processed cormorant dataset.  Extends PyTorch Dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    included_species : tensor of scalars, optional
        Atomic species to include in ?????.  If None, uses all species.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    normalize : bool, optional
        ????? IS THIS USED?
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.

    """
    def __init__(self, data, included_species=None, num_pts=-1, normalize=True, shuffle=True, subtract_thermo=True):

        self.data = data

        if num_pts < 0:
            self.num_pts = len(data['charges'])
        else:
            if num_pts > len(data['charges']):
                logging.warning('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(data['charges'])))
                self.num_pts = len(data['charges'])
            else:
                self.num_pts = num_pts

        # If included species is not specified
        if included_species is None:
            included_species = torch.unique(self.data['charges'], sorted=True)
            if included_species[0] == 0:
                included_species = included_species[1:]

        if subtract_thermo:
            thermo_targets = [key.split('_')[0] for key in data.keys() if key.endswith('_thermo')]
            if len(thermo_targets) == 0:
                logging.warning('No thermochemical targets included! Try reprocessing dataset with --force-download!')
            else:
                logging.info('Removing thermochemical energy from targets {}'.format(' '.join(thermo_targets)))
            for key in thermo_targets:
                data[key] -= data[key + '_thermo'].to(data[key].dtype)

        self.included_species = included_species

        self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)

        self.num_species = len(included_species)
        self.max_charge = max(included_species)

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}

        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()

        if shuffle:
            self.perm = torch.randperm(len(data['charges']))[:self.num_pts]
        else:
            self.perm = None

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        return {key: val[idx] for key, val in self.data.items()}



def initialize_datasets(args, datadir, dataset, subset=None, splits=None,
                        force_download=False, subtract_thermo=False,
                        remove_h=False):
    """
    Initialize datasets.

    Parameters
    ----------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : str, optional
        TODO: DELETE THIS ENTRY
    force_download : bool, optional
        If true, forces a fresh download of the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.
    remove_h: bool, optional
        If True, remove hydrogens from the dataset
    Returns
    -------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datasets : dict
        Dictionary of processed dataset objects (see ????? for more information).
        Valid keys are "train", "test", and "valid"[ate].  Each associated value
    num_species : int
        Number of unique atomic species in the dataset.
    max_charge : pytorch.Tensor
        Largest atomic number for the dataset.

    Notes
    -----
    TODO: Delete the splits argument.
    """
    # Set the number of points based upon the arguments
    num_pts = {'train': args.num_train,
               'test': args.num_test, 'valid': args.num_valid}

    # Download and process dataset. Returns datafiles.
    datafiles = prepare_dataset(
        datadir, 'qm9', subset, splits, force_download=force_download)

    # Load downloaded/processed datasets
    datasets = {}
    for split, datafile in datafiles.items():
        with np.load(datafile) as f:
            datasets[split] = {key: torch.from_numpy(
                val) for key, val in f.items()}

    if dataset != 'qm9':
        np.random.seed(42)
        fixed_perm = np.random.permutation(len(datasets['train']['num_atoms']))
        if dataset == 'qm9_second_half':
            sliced_perm = fixed_perm[len(datasets['train']['num_atoms'])//2:]
        elif dataset == 'qm9_first_half':
            sliced_perm = fixed_perm[0:len(datasets['train']['num_atoms']) // 2]
        else:
            raise Exception('Wrong dataset name')
        for key in datasets['train']:
            datasets['train'][key] = datasets['train'][key][sliced_perm]

    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]
               ), 'Datasets must have same set of keys!'

    # TODO: remove hydrogens here if needed
    if remove_h:
        print('Removing h')
        for key, dataset in datasets.items():
            pos = dataset['positions']
            charges = dataset['charges']
            num_atoms = dataset['num_atoms']

            # Check that charges corresponds to real atoms
            assert torch.sum(num_atoms != torch.sum(charges > 0, dim=1)) == 0

            mask = dataset['charges'] > 1
            new_positions = torch.zeros_like(pos)
            new_charges = torch.zeros_like(charges)
            for i in range(new_positions.shape[0]):
                m = mask[i]
                p = pos[i][m]   # positions to keep
                p = p - torch.mean(p, dim=0)    # Center the new positions
                c = charges[i][m]   # Charges to keep
                n = torch.sum(m)
                new_positions[i, :n, :] = p
                new_charges[i, :n] = c

            dataset['positions'] = new_positions
            dataset['charges'] = new_charges
            dataset['num_atoms'] = torch.sum(dataset['charges'] > 0, dim=1)

    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets, ignore_check=False)

    # Now initialize MolecularDataset based upon loaded data
    datasets = {split: ProcessedDataset(data, num_pts=num_pts.get(
        split, -1), included_species=all_species, subtract_thermo=subtract_thermo) for split, data in datasets.items()}

    # Now initialize MolecularDataset based upon loaded data

    # Check that all datasets have the same included species:
    assert(len(set(tuple(data.included_species.tolist()) for data in datasets.values())) ==
           1), 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})

    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge

    # Now, update the number of training/test/validation sets in args
    args.num_train = datasets['train'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts

    return args, datasets, num_species, max_charge


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets

def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]

class PreprocessQM9:
    def __init__(self, load_charges=True):
        self.load_charges = load_charges

    def add_trick(self, trick):
        self.tricks.append(trick)

    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for cormorant

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : dict of Pytorch tensors
            The collated data.
        """
        batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

        to_keep = (batch['charges'].sum(0) > 0)

        batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

        atom_mask = batch['charges'] > 0
        batch['atom_mask'] = atom_mask

        #Obtain edges
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        #mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask

        #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        if self.load_charges:
            batch['charges'] = batch['charges'].unsqueeze(2)
        else:
            batch['charges'] = torch.zeros(0)
        return batch


# def retrieve_dataloaders(cfg):
#     if 'qm9' in cfg.dataset:
#         batch_size = cfg.batch_size
#         num_workers = cfg.num_workers
#         filter_n_atoms = cfg.filter_n_atoms
#         # Initialize dataloader
# 
#         # args = init_argparse('qm9')
# 
#         # data_dir = cfg.data_root_dir
#         cfg, datasets, num_species, charge_scale = initialize_datasets(cfg, cfg.datadir, cfg.dataset,
#                                                                         subtract_thermo=args.subtract_thermo,
#                                                                         force_download=args.force_download,
#                                                                         remove_h=cfg.remove_h)
#         qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
#                      'lumo': 27.2114}
# 
#         for dataset in datasets.values():
#             dataset.convert_units(qm9_to_eV)
# 
#         if filter_n_atoms is not None:
#             print("Retrieving molecules with only %d atoms" % filter_n_atoms)
#             datasets = filter_atoms(datasets, filter_n_atoms)
# 
#         # Construct PyTorch dataloaders from datasets
#         preprocess = PreprocessQM9(load_charges=cfg.include_charges)
#         dataloaders = {split: DataLoader(dataset,
#                                          batch_size=batch_size,
#                                          shuffle=args.shuffle if (split == 'train') else False,
#                                          num_workers=num_workers,
#                                          collate_fn=preprocess.collate_fn)
#                              for split, dataset in datasets.items()}
#     elif 'geom' in cfg.dataset:
#         raise NotImplementedError
#         # import build_geom_dataset
#         # from configs.datasets_config import get_dataset_info
#         # data_file = './data/geom/geom_drugs_30.npy'
#         # dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)
# 
#         # # Retrieve QM9 dataloaders
#         # split_data = build_geom_dataset.load_split_data(data_file,
#         #                                                 val_proportion=0.1,
#         #                                                 test_proportion=0.1,
#         #                                                 filter_size=cfg.filter_molecule_size)
#         # transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
#         #                                                   cfg.include_charges,
#         #                                                   cfg.device,
#         #                                                   cfg.sequential)
#         # dataloaders = {}
#         # for key, data_list in zip(['train', 'val', 'test'], split_data):
#         #     dataset = build_geom_dataset.GeomDrugsDataset(data_list,
#         #                                                   transform=transform)
#         #     shuffle = (key == 'train') and not cfg.sequential
# 
#         #     # Sequential dataloading disabled for now.
#         #     dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
#         #         sequential=cfg.sequential, dataset=dataset,
#         #         batch_size=cfg.batch_size,
#         #         shuffle=shuffle)
#         # del split_data
#         # charge_scale = None
#     else:
#         raise ValueError(f'Unknown dataset {cfg.dataset}')
# 
#     return dataloaders, charge_scale


qm9_with_h = {
    'name': 'qm9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'n_nodes': {22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060,
                15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362,
                8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 1},
    'max_n_nodes': 29,
    'atom_types': {1: 635559, 2: 101476, 0: 923537, 3: 140202, 4: 2323},
    'distances': [903054, 307308, 111994, 57474, 40384, 29170, 47152, 414344, 2202212, 573726,
                  1490786, 2970978, 756818, 969276, 489242, 1265402, 4587994, 3187130, 2454868, 2647422,
                  2098884,
                  2001974, 1625206, 1754172, 1620830, 1710042, 2133746, 1852492, 1415318, 1421064, 1223156,
                  1322256,
                  1380656, 1239244, 1084358, 981076, 896904, 762008, 659298, 604676, 523580, 437464, 413974,
                  352372,
                  291886, 271948, 231328, 188484, 160026, 136322, 117850, 103546, 87192, 76562, 61840,
                  49666, 43100,
                  33876, 26686, 22402, 18358, 15518, 13600, 12128, 9480, 7458, 5088, 4726, 3696, 3362, 3396,
                  2484,
                  1988, 1490, 984, 734, 600, 456, 482, 378, 362, 168, 124, 94, 88, 52, 44, 40, 18, 16, 8, 6,
                  2,
                  0, 0, 0, 0,
                  0,
                  0, 0],
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77, 0.77],
    'with_h': True}
    # 'bond1_radius': {'H': 31, 'C': 76, 'N': 71, 'O': 66, 'F': 57},
    # 'bond1_stdv': {'H': 5, 'C': 2, 'N': 2, 'O': 2, 'F': 3},
    # 'bond2_radius': {'H': -1000, 'C': 67, 'N': 60, 'O': 57, 'F': 59},
    # 'bond3_radius': {'H': -1000, 'C': 60, 'N': 54, 'O': 53, 'F': 53}}

qm9_without_h = {
    'name': 'qm9',
    'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'F': 3},
    'atom_decoder': ['C', 'N', 'O', 'F'],
    'max_n_nodes': 29,
    'n_nodes': {9: 83366, 8: 13625, 7: 2404, 6: 475, 5: 91, 4: 25, 3: 7, 1: 2, 2: 5},
    'atom_types': {0: 635559, 2: 140202, 1: 101476, 3: 2323},
    'distances': [594, 1232, 3706, 4736, 5478, 9156, 8762, 13260, 45674, 174676, 469292,
                    1182942, 126722, 25768, 28532, 51696, 232014, 299916, 686590, 677506,
                    379264, 162794, 158732, 156404, 161742, 156486, 236176, 310918, 245558,
                    164688, 98830, 81786, 89318, 91104, 92788, 83772, 81572, 85032, 56296,
                    32930, 22640, 24124, 24010, 22120, 19730, 21968, 18176, 12576, 8224,
                    6772,
                    3906, 4416, 4306, 4110, 3700, 3592, 3134, 2268, 774, 674, 514, 594, 622,
                    672, 642, 472, 300, 170, 104, 48, 54, 78, 78, 56, 48, 36, 26, 4, 2, 4,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'colors_dic': ['C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.77, 0.77, 0.77, 0.77],
    'with_h': False}
    # 'bond1_radius': {'C': 76, 'N': 71, 'O': 66, 'F': 57},
    # 'bond1_stdv': {'C': 2, 'N': 2, 'O': 2, 'F': 3},
    # 'bond2_radius': {'C': 67, 'N': 60, 'O': 57, 'F': 59},
    # 'bond3_radius': {'C': 60, 'N': 54, 'O': 53, 'F': 53}}


qm9_second_half = {
    'name': 'qm9_second_half',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'n_nodes': {19: 6944, 12: 845, 20: 4794, 21: 4962, 27: 132, 25: 754, 18: 6695, 14: 2587, 15: 3865, 22: 1701, 17: 6461, 16: 5344, 23: 2380, 13: 1541, 24: 267, 10: 178, 7: 7, 11: 412, 8: 25, 9: 62, 29: 15, 26: 17, 4: 3, 3: 1, 6: 5, 5: 3},
    'atom_types': {1: 317604, 2: 50852, 3: 70033, 0: 461622, 4: 1164},
    'distances': [457374, 153688, 55626, 28284, 20414, 15010, 24412, 208012, 1105440, 285830, 748876, 1496486, 384178, 484194, 245688, 635534, 2307642, 1603762, 1231044, 1329758, 1053612, 1006742, 813504, 880670, 811616, 855082, 1066434, 931672, 709810, 711032, 608446, 660538, 692382, 619084, 544200, 490740, 450576, 380662, 328150, 303008, 263888, 218820, 207414, 175452, 145636, 135646, 116184, 94622, 80358, 68230, 58706, 51216, 44020, 38212, 30492, 24886, 21210, 17270, 13056, 11156, 9082, 7534, 6958, 6060, 4632, 3760, 2500, 2342, 1816, 1726, 1768, 1102, 974, 670, 474, 446, 286, 246, 242, 156, 176, 90, 66, 66, 38, 28, 24, 14, 10, 2, 6, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77, 0.77],
    'max_n_nodes': 29,
    'with_h': True}
    # 'bond1_radius': {'H': 31, 'C': 76, 'N': 71, 'O': 66, 'F': 57},
    # 'bond1_stdv': {'H': 5, 'C': 2, 'N': 2, 'O': 2, 'F': 3},
    # 'bond2_radius': {'H': -1000, 'C': 67, 'N': 60, 'O': 57, 'F': 59},
    # 'bond3_radius': {'H': -1000, 'C': 60, 'N': 54, 'O': 53, 'F': 53}}


geom_with_h = {
    'name': 'geom',
    'atom_encoder': {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
    'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15},
    'atomic_nb': [1,  5,  6,  7,  8,  9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83],
    'atom_decoder': ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi'],
    'max_n_nodes': 181,
    'n_nodes': {3: 1, 4: 3, 5: 9, 6: 2, 7: 8, 8: 23, 9: 23, 10: 50, 11: 109, 12: 168, 13: 280, 14: 402, 15: 583, 16: 597,
                17: 949, 18: 1284, 19: 1862, 20: 2674, 21: 3599, 22: 6109, 23: 8693, 24: 13604, 25: 17419, 26: 25672,
                27: 31647, 28: 43809, 29: 56697, 30: 70400, 31: 82655, 32: 104100, 33: 122776, 34: 140834, 35: 164888,
                36: 185451, 37: 194541, 38: 218549, 39: 231232, 40: 243300, 41: 253349, 42: 268341, 43: 272081,
                44: 276917, 45: 276839, 46: 274747, 47: 272126, 48: 262709, 49: 250157, 50: 244781, 51: 228898,
                52: 215338, 53: 203728, 54: 191697, 55: 180518, 56: 163843, 57: 152055, 58: 136536, 59: 120393,
                60: 107292, 61: 94635, 62: 83179, 63: 68384, 64: 61517, 65: 48867, 66: 37685, 67: 32859, 68: 27367,
                69: 20981, 70: 18699, 71: 14791, 72: 11921, 73: 9933, 74: 9037, 75: 6538, 76: 6374, 77: 4036, 78: 4189,
                79: 3842, 80: 3277, 81: 2925, 82: 1843, 83: 2060, 84: 1394, 85: 1514, 86: 1357, 87: 1346, 88: 999,
                89: 300, 90: 390, 91: 510, 92: 510, 93: 240, 94: 721, 95: 360, 96: 360, 97: 390, 98: 330, 99: 540,
                100: 258, 101: 210, 102: 60, 103: 180, 104: 206, 105: 60, 106: 390, 107: 180, 108: 180, 109: 150,
                110: 120, 111: 360, 112: 120, 113: 210, 114: 60, 115: 30, 116: 210, 117: 270, 118: 450, 119: 240,
                120: 228, 121: 120, 122: 30, 123: 420, 124: 240, 125: 210, 126: 158, 127: 180, 128: 60, 129: 30,
                130: 120, 131: 30, 132: 120, 133: 60, 134: 240, 135: 169, 136: 240, 137: 30, 138: 270, 139: 180,
                140: 270, 141: 150, 142: 60, 143: 60, 144: 240, 145: 180, 146: 150, 147: 150, 148: 90, 149: 90,
                151: 30, 152: 60, 155: 90, 159: 30, 160: 60, 165: 30, 171: 30, 175: 30, 176: 60, 181: 30},
    'atom_types':{0: 143905848, 1: 290, 2: 129988623, 3: 20266722, 4: 21669359, 5: 1481844, 6: 1,
                  7: 250, 8: 36290, 9: 3999872, 10: 1224394, 11: 4, 12: 298702, 13: 5377, 14: 13, 15: 34},
    'colors_dic': ['#FFFFFF99',
                   'C2', 'C7', 'C0', 'C3', 'C1', 'C5',
                   'C6', 'C4', 'C8', 'C9', 'C10',
                   'C11', 'C12', 'C13', 'C14'],
    'radius_dic': [0.3, 0.6, 0.6, 0.6, 0.6,
                   0.6, 0.6, 0.6, 0.6, 0.6,
                   0.6, 0.6, 0.6, 0.6, 0.6,
                   0.6],
    'with_h': True}


geom_no_h = {
    'name': 'geom',
    'atom_encoder': {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Al': 5, 'Si': 6, 'P': 7, 'S': 8, 'Cl': 9, 'As': 10,
                     'Br': 11, 'I': 12, 'Hg': 13, 'Bi': 14},
    'atomic_nb': [5,  6,  7,  8,  9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83],
    'atom_decoder': ['B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi'],
    'max_n_nodes': 91,
    'n_nodes': {1: 3, 2: 5, 3: 8, 4: 89, 5: 166, 6: 370, 7: 613, 8: 1214, 9: 1680, 10: 3315, 11: 5115, 12: 9873,
                13: 15422, 14: 28088, 15: 50643, 16: 82299, 17: 124341, 18: 178417, 19: 240446, 20: 308209, 21: 372900,
                22: 429257, 23: 477423, 24: 508377, 25: 522385, 26: 522000, 27: 507882, 28: 476702, 29: 426308,
                30: 375819, 31: 310124, 32: 255179, 33: 204441, 34: 149383, 35: 109343, 36: 71701, 37: 44050,
                38: 31437, 39: 20242, 40: 14971, 41: 10078, 42: 8049, 43: 4476, 44: 3130, 45: 1736, 46: 2030,
                47: 1110, 48: 840, 49: 750, 50: 540, 51: 810, 52: 591, 53: 453, 54: 540, 55: 720, 56: 300, 57: 360,
                58: 714, 59: 390, 60: 519, 61: 210, 62: 449, 63: 210, 64: 289, 65: 589, 66: 227, 67: 180, 68: 330,
                69: 330, 70: 150, 71: 60, 72: 210, 73: 60, 74: 180, 75: 120, 76: 30, 77: 150, 78: 30, 79: 60, 82: 60,
                85: 60, 86: 6, 87: 60, 90: 60, 91: 30},
    'atom_types': {0: 290, 1: 129988623, 2: 20266722, 3: 21669359, 4: 1481844, 5: 1, 6: 250, 7: 36290, 8: 3999872,
                   9: 1224394, 10: 4, 11: 298702, 12: 5377, 13: 13, 14: 34},
    'colors_dic': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14'],
    'radius_dic': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    'with_h': False}


def get_dataset_info(dataset_name, remove_h):
    if dataset_name == 'qm9':
        if not remove_h:
            return qm9_with_h
        else:
            return qm9_without_h
    elif dataset_name == 'geom':
        if not remove_h:
            return geom_with_h
        else:
            raise Exception('Missing config for %s without hydrogens' % dataset_name)
    elif dataset_name == 'qm9_second_half':
        if not remove_h:
            return qm9_second_half
        else:
            raise Exception('Missing config for %s without hydrogens' % dataset_name)
    else:
        raise Exception("Wrong dataset %s" % dataset_name)


def compute_mean_mad(dataloaders, properties, dataset_name):
    if dataset_name == 'qm9':
        return compute_mean_mad_from_dataloader(dataloaders['train'], properties)
    elif dataset_name == 'qm9_second_half' or dataset_name == 'qm9_second_half':
        return compute_mean_mad_from_dataloader(dataloaders['valid'], properties)
    else:
        raise Exception('Wrong dataset name')


def compute_mean_mad_from_dataloader(dataloader, properties):
    property_norms = {}
    for property_key in properties:
        values = dataloader.dataset.data[property_key]
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        property_norms[property_key] = {}
        property_norms[property_key]['mean'] = mean
        property_norms[property_key]['mad'] = mad
    return property_norms

def prepare_context(conditioning, minibatch, property_norms):
    batch_size, n_nodes, _ = minibatch['positions'].size()
    node_mask = minibatch['atom_mask'].unsqueeze(2)
    context_node_nf = 0
    context_list = []
    for key in conditioning:
        properties = minibatch[key]
        properties = (properties - property_norms[key]['mean']) / property_norms[key]['mad']
        if len(properties.size()) == 1:
            # Global feature.
            assert properties.size() == (batch_size,)
            reshaped = properties.view(batch_size, 1, 1).repeat(1, n_nodes, 1)
            context_list.append(reshaped)
            context_node_nf += 1
        elif len(properties.size()) == 2 or len(properties.size()) == 3:
            # Node feature.
            assert properties.size()[:2] == (batch_size, n_nodes)

            context_key = properties

            # Inflate if necessary.
            if len(properties.size()) == 2:
                context_key = context_key.unsqueeze(2)

            context_list.append(context_key)
            context_node_nf += context_key.size(2)
        else:
            raise ValueError('Invalid tensor size, more than 3 axes.')
    # Concatenate
    context = torch.cat(context_list, dim=2)
    # Mask disabled nodes!
    context = context * node_mask
    assert context.size(2) == context_node_nf
    return context






# From --------------------- qm9/analyze --------------------------------------




try:
    from rdkit import Chem
    # from training.rdkit_functions import BasicMolecularMetrics
    use_rdkit = True
    print('using rdkit')
    bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC]
except ModuleNotFoundError:
    use_rdkit = False
    print('not using rdkit')
# use_rdkit = False







def analyze_stability_for_molecules(molecule_list, dataset_info, dataloaders):
    one_hot = molecule_list['one_hot']
    x = molecule_list['x']
    node_mask = molecule_list['node_mask']

    if isinstance(node_mask, torch.Tensor):
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [torch.sum(m) for m in node_mask]

    n_samples = len(x)

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    processed_list = []

    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:int(atomsxmol[i])]
        pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))

    for mol in processed_list:
        pos, atom_type = mol
        validity_results = check_stability(pos, atom_type, dataset_info)

        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])

    # Validity
    fraction_mol_stable = molecule_stable / float(n_samples)
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
    }

    if use_rdkit:
        # raise NotImplementedError
        metrics = BasicMolecularMetrics(dataset_info, dataloaders)
        rdkit_metrics = metrics.evaluate(processed_list)
        #print("Unique molecules:", rdkit_metrics[1])
        return validity_dict, rdkit_metrics
    else:
        return validity_dict, None

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3,
                 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2],
                 'Bi': [3, 5]}


def check_stability(positions, atom_type, dataset_info, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info['atom_decoder']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_first_half':
                order = get_bond_order(atom1, atom2, dist)
            elif dataset_info['name'] == 'geom':
                raise NotImplementedError
                # order = bond_analyze.geom_predictor(
                #     (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)

# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H':  119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186}}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}
margin1, margin2, margin3 = 10, 5, 3

def get_bond_order(atom1, atom2, distance, check_exists=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:

        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3        # Triple
                return 2            # Double
        return 1                # Single
    return 0                    # No bond

# ----------------------- end from qm9/analyze --------------------------------





def get_cfg():

    # These are from the EGNN repo outputs/edm_qm9/args.pickle
    args = Namespace(exp_name='polynomial_2_final_0', model='egnn_dynamics',
            probabilistic_model='diffusion', diffusion_steps=1000,
            diffusion_noise_schedule='polynomial_2', diffusion_noise_precision=1e-05,
            diffusion_loss_type='l2', n_epochs=3000, batch_size=64, lr=0.0001,
            brute_force=False, actnorm=True, break_train_epoch=False, dp=True,
            condition_time=True, clip_grad=True, trace='hutch', n_layers=9,
            inv_sublayers=1, nf=256, tanh=True, attention=True, norm_constant=1,
            sin_embedding=False, ode_regularization=0.001, dataset='qm9',
            datadir='qm9/temp', filter_n_atoms=None, dequantization='argmax_variational',
            n_report_steps=1, wandb_usr=None, no_wandb=False, online=True, no_cuda=False,
            save_model=True, generate_epochs=1, num_workers=0, test_epochs=20,
            data_augmentation=False, conditioning=[], resume=None, start_epoch=0,
            ema_decay=0.9999, augment_noise=0, n_stability_samples=1000,
            normalize_factors=[1, 4, 10], remove_h=False, include_charges=True,
            visualize_every_batch=100000000.0, normalization_factor=1, cuda=True,
            context_node_nf=0)

    # This is the suggested run command from which we can verify some args
    # python main_qm9.py --n_epochs 3000 --exp_name edm_qm9 --n_stability_samples 1000 
    # --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5
    # --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256
    # --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999```


    # --------------- From main_qm9.py ----------------------------------------

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1.0
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'
    # -------------------------------------------------------------------------
    return args


# # Retrieve QM9 dataloaders
# dataloaders, charge_scale = retrieve_dataloaders(args)
# 
# 
# 
# data_dummy = next(iter(dataloaders['train']))
# 
# 
# if len(args.conditioning) > 0:
#     print(f'Conditioning on {args.conditioning}')
#     property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
#     context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
#     context_node_nf = context_dummy.size(2)
# else:
#     context_node_nf = 0
#     property_norms = None
# 
# args.context_node_nf = context_node_nf
#         
# 
# dataset_info = qm9_with_h


class BoolArg(argparse.Action):
    """
    Take an argparse argument that is either a boolean or a string and return a boolean.
    """
    def __init__(self, default=None, nargs=None, *args, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")

        # Set default
        if default is None:
            raise ValueError("Default must be set!")

        default = _arg_to_bool(default)

        super().__init__(*args, default=default, nargs='?', **kwargs)

    def __call__(self, parser, namespace, argstring, option_string):

        if argstring is not None:
            # If called with an argument, convert to bool
            argval = _arg_to_bool(argstring)
        else:
            # BoolArg will invert default option
            argval = True

        setattr(namespace, self.dest, argval)

def _arg_to_bool(arg):
    # Convert argument to boolean

    if type(arg) is bool:
        # If argument is bool, just return it
        return arg

    elif type(arg) is str:
        # If string, convert to true/false
        arg = arg.lower()
        if arg in ['true', 't', '1']:
            return True
        elif arg in ['false', 'f', '0']:
            return False
        else:
            return ValueError('Could not parse a True/False boolean')
    else:
        raise ValueError('Input must be boolean or string! {}'.format(type(arg)))


# From https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end


def setup_shared_args(parser):
    """
    Sets up the argparse object for the qm9 dataset
    
    Parameters 
    ----------
    parser : :class:`argparse.ArgumentParser`
        Argument Parser with arguments.
    
    Parameters 
    ----------
    parser : :class:`argparse.ArgumentParser`
        The same Argument Parser, now with more arguments.
    """
    # Optimizer options
    parser.add_argument('--num-epoch', type=int, default=255, metavar='N',
                        help='number of epochs to train (default: 511)')
    parser.add_argument('--batch-size', '-bs', type=int, default=25, metavar='N',
                        help='Mini-batch size (default: 25)')
    parser.add_argument('--alpha', type=float, default=0.9, metavar='N',
                        help='Value of alpha to use for exponential moving average of training loss. (default: 0.9)')

    parser.add_argument('--weight-decay', type=float, default=0, metavar='N',
                        help='Set the weight decay used in optimizer (default: 0)')
    parser.add_argument('--cutoff-decay', type=float, default=0, metavar='N',
                        help='Set the weight decay used in optimizer for learnable radial cutoffs (default: 0)')
    parser.add_argument('--lr-init', type=float, default=1e-3, metavar='N',
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--lr-final', type=float, default=1e-5, metavar='N',
                        help='Final (held) learning rate (default: 1e-5)')
    parser.add_argument('--lr-decay', type=int, default=inf, metavar='N',
                        help='Timescale over which to decay the learning rate (default: inf)')
    parser.add_argument('--lr-decay-type', type=str, default='cos', metavar='str',
                        help='Type of learning rate decay. (cos | linear | exponential | pow | restart) (default: cos)')
    parser.add_argument('--lr-minibatch', '--lr-mb', action=BoolArg, default=True,
                        help='Decay learning rate every minibatch instead of epoch.')
    parser.add_argument('--sgd-restart', type=int, default=-1, metavar='int',
                        help='Restart SGD optimizer every (lr_decay)^p epochs, where p=sgd_restart. (-1 to disable) (default: -1)')

    parser.add_argument('--optim', type=str, default='amsgrad', metavar='str',
                        help='Set optimizer. (SGD, AMSgrad, Adam, RMSprop)')

    # Dataloader and randomness options
    parser.add_argument('--shuffle', action=BoolArg, default=True,
                        help='Shuffle minibatches.')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='Set random number seed. Set to -1 to set based upon clock.')

    # Saving and logging options
    parser.add_argument('--save', action=BoolArg, default=True,
                        help='Save checkpoint after each epoch. (default: True)')
    parser.add_argument('--load', action=BoolArg, default=False,
                        help='Load from previous checkpoint. (default: False)')

    parser.add_argument('--test', action=BoolArg, default=True,
                        help='Perform automated network testing. (Default: True)')

    parser.add_argument('--log-level', type=str, default='info',
                        help='Logging level to output')

    parser.add_argument('--textlog', action=BoolArg, default=True,
                        help='Log a summary of each mini-batch to a text file.')

    parser.add_argument('--predict', action=BoolArg, default=True,
                        help='Save predictions. (default)')

    ### Arguments for files to save things to
    # Job prefix is used to name checkpoint/best file
    parser.add_argument('--prefix', '--jobname', type=str, default='nosave',
                        help='Prefix to set load, save, and logfile. (default: nosave)')

    # Allow to manually specify file to load
    parser.add_argument('--loadfile', type=str, default='',
                        help='Set checkpoint file to load. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to save model checkpoint to
    parser.add_argument('--checkfile', type=str, default='',
                        help='Set checkpoint file to save checkpoints to. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to best model checkpoint to
    parser.add_argument('--bestfile', type=str, default='',
                        help='Set checkpoint file to best model to. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to save logging information to
    parser.add_argument('--logfile', type=str, default='',
                        help='Duplicate logging.info output to logfile. Set to empty string to generate from prefix. (default: (empty))')
    # Filename to save predictions to
    parser.add_argument('--predictfile', type=str, default='',
                        help='Save predictions to file. Set to empty string to generate from prefix. (default: (empty))')

    # Working directory to place all files
    parser.add_argument('--workdir', type=str, default='./',
                        help='Working directory as a default location for all files. (default: ./)')
    # Directory to place logging information
    parser.add_argument('--logdir', type=str, default='log/',
                        help='Directory to place log and savefiles. (default: log/)')
    # Directory to place saved models
    parser.add_argument('--modeldir', type=str, default='model/',
                        help='Directory to place log and savefiles. (default: model/)')
    # Directory to place model predictions
    parser.add_argument('--predictdir', type=str, default='predict/',
                        help='Directory to place log and savefiles. (default: predict/)')
    # Directory to read and save data from
    parser.add_argument('--datadir', type=str, default='qm9/temp',
                        help='Directory to look up data from. (default: data/)')

    # Dataset options
    parser.add_argument('--num-train', type=int, default=-1, metavar='N',
                        help='Number of samples to train on. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--num-valid', type=int, default=-1, metavar='N',
                        help='Number of validation samples to use. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--num-test', type=int, default=-1, metavar='N',
                        help='Number of test samples to use. Set to -1 to use entire dataset. (default: -1)')

    parser.add_argument('--force-download', action=BoolArg, default=False,
                        help='Force download and processing of dataset.')

    # Computation options
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Use CUDA (default)')
    parser.add_argument('--no-cuda', '--cpu', dest='cuda', action='store_false',
                        help='Use CPU')
    parser.set_defaults(cuda=True)

    parser.add_argument('--float', dest='dtype', action='store_const', const='float',
                        help='Use floats.')
    parser.add_argument('--double', dest='dtype', action='store_const', const='double',
                        help='Use doubles.')
    parser.set_defaults(dtype='float')

    parser.add_argument('--num-workers', type=int, default=8,
                        help='Set number of workers in dataloader. (Default: 1)')

    # Model options
    parser.add_argument('--num-cg-levels', type=int, default=4, metavar='N',
                        help='Number of CG levels (default: 4)')

    parser.add_argument('--maxl', nargs='*', type=int, default=[3], metavar='N',
                        help='Cutoff in CG operations (default: [3])')
    parser.add_argument('--max-sh', nargs='*', type=int, default=[3], metavar='N',
                        help='Number of spherical harmonic powers to use (default: [3])')
    parser.add_argument('--num-channels', nargs='*', type=int, default=[10], metavar='N',
                        help='Number of channels to allow after mixing (default: [10])')
    parser.add_argument('--level-gain', nargs='*', type=float, default=[10.], metavar='N',
                        help='Gain at each level (default: [10.])')

    parser.add_argument('--charge-power', type=int, default=2, metavar='N',
                        help='Maximum power to take in one-hot (default: 2)')

    parser.add_argument('--hard-cutoff', dest='hard_cut_rad',
                        type=float, default=1.73, nargs='*', metavar='N',
                        help='Radius of HARD cutoff in Angstroms (default: 1.73)')
    parser.add_argument('--soft-cutoff', dest='soft_cut_rad', type=float,
                        default=1.73, nargs='*', metavar='N',
                        help='Radius of SOFT cutoff in Angstroms (default: 1.73)')
    parser.add_argument('--soft-width', dest='soft_cut_width',
                        type=float, default=0.2, nargs='*', metavar='N',
                        help='Width of SOFT cutoff in Angstroms (default: 0.2)')
    parser.add_argument('--cutoff-type', '--cutoff', type=str, default=['learn'], nargs='*', metavar='str',
                        help='Types of cutoffs to include')

    parser.add_argument('--basis-set', '--krange', type=int, default=[3, 3], nargs=2, metavar='N',
                        help='Radial function basis set (m, n) size (default: [3, 3])')

    # TODO: Update(?)
    parser.add_argument('--weight-init', type=str, default='rand', metavar='str',
                        help='Weight initialization function to use (default: rand)')

    parser.add_argument('--input', type=str, default='linear',
                        help='Function to apply to process l0 input (linear | MPNN) default: linear')
    parser.add_argument('--num-mpnn-levels', type=int, default=1,
                        help='Number levels to use in input featurization MPNN. (default: 1)')
    parser.add_argument('--top', '--output', type=str, default='linear',
                        help='Top function to use (linear | PMLP) default: linear')

    parser.add_argument('--gaussian-mask', action='store_true',
                        help='Use gaussian mask instead of sigmoid mask.')

    parser.add_argument('--edge-cat', action='store_true',
                        help='Concatenate the scalars from different \ell in the dot-product-matrix part of the edge network.')
    parser.add_argument('--target', type=str, default='',
                        help='Learning target for a dataset (such as qm9) with multiple options.')

    return parser

def setup_argparse(dataset):
    """
    Sets up the argparse object for a specific dataset.

    Parameters
    ----------
    dataset : :class:`str`
        Dataset being used.  Currently MD17 and QM9 are supported.

    Returns
    -------
    parser : :class:`argparse.ArgumentParser`
        Argument Parser with arguments.
    """
    parser = argparse.ArgumentParser(description='Cormorant network options for the md17 dataset.')
    parser = setup_shared_args(parser)
    if dataset == "md17":
        parser.add_argument('--subset', '--molecule', type=str, default='',
                            help='Subset/molecule on data with subsets (such as md17).')
    elif dataset == "qm9":
        parser.add_argument('--subtract-thermo', action=BoolArg, default=True,
                            help='Subtract thermochemical energy from relvant learning targets in QM9 dataset.')
    else:
        raise ValueError("Dataset is not recognized")
    return parser

def init_argparse(dataset):
    """
    Reads in the arguments for the script for a given dataset.

    Parameters
    ----------
    dataset : :class:`str`
        Dataset being used.  Currently 'md17' and 'qm9' are supported.

    Returns
    -------
    args : :class:`Namespace`
        Namespace with a dictionary of arguments where the key is the name of
        the argument and the item is the input value.
    """

    parser = setup_argparse(dataset)
    args = parser.parse_args([])
    d = vars(args)
    d['dataset'] = dataset

    return args


# ---------------------- From visualize.py -----------------------------


def draw_sphere(ax, x, y, z, size, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) * 0.8  # Correct for matplotlib.
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))
    # for i in range(2):
    #    ax.plot_surface(x+random.randint(-5,5), y+random.randint(-5,5), z+random.randint(-5,5),  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color, linewidth=0,
                    alpha=alpha)
    # # calculate vectors for "vertical" circle
    # a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
    # b = np.array([0, 1, 0])
    # b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (
    #             1 - np.cos(rot))
    # ax.plot(np.sin(u), np.cos(u), 0, color='k', linestyle='dashed')
    # horiz_front = np.linspace(0, np.pi, 100)
    # ax.plot(np.sin(horiz_front), np.cos(horiz_front), 0, color='k')
    # vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
    # ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u),
    #         a[2] * np.sin(u) + b[2] * np.cos(u), color='k', linestyle='dashed')
    # ax.plot(a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front),
    #         b[1] * np.cos(vert_front),
    #         a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front), color='k')
    #
    # ax.view_init(elev=elev, azim=0)


def plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color,
                  dataset_info):
    # draw_sphere(ax, 0, 0, 0, 1)
    # draw_sphere(ax, 1, 1, 1, 1)

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Hydrogen, Carbon, Nitrogen, Oxygen, Flourine

    # ax.set_facecolor((1.0, 0.47, 0.42))
    colors_dic = np.array(dataset_info['colors_dic'])
    radius_dic = np.array(dataset_info['radius_dic'])
    area_dic = 1500 * radius_dic ** 2
    # areas_dic = sizes_dic * sizes_dic * 3.1416

    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]

    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c, alpha)
    else:
        ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha,
                   c=colors)  # , linewidths=2, edgecolors='#FFFFFF')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = dataset_info['atom_decoder'][atom_type[i]], \
                           dataset_info['atom_decoder'][atom_type[j]]
            s = sorted((atom_type[i], atom_type[j]))
            pair = (dataset_info['atom_decoder'][s[0]],
                    dataset_info['atom_decoder'][s[1]])
            if 'qm9' in dataset_info['name']:
                draw_edge_int = get_bond_order(atom1, atom2, dist)
                line_width = (3 - 2) * 2 * 2
            elif dataset_info['name'] == 'geom':
                raise NotImplementedError
                # draw_edge_int = bond_analyze.geom_predictor(pair, dist)
                # # Draw edge outputs 1 / -1 value, convert to True / False.
                # line_width = 2
            else:
                raise Exception('Wrong dataset_info name')
            draw_edge = draw_edge_int > 0
            if draw_edge:
                if draw_edge_int == 4:
                    linewidth_factor = 1.5
                else:
                    # linewidth_factor = draw_edge_int  # Prop to number of
                    # edges.
                    linewidth_factor = 1
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                        linewidth=line_width * linewidth_factor,
                        c=hex_bg_color, alpha=alpha)


def plot_data3d(positions, atom_type, dataset_info, camera_elev=0, camera_azim=0, save_path=None, spheres_3d=False,
                bg='black', alpha=1.):
    # positions should be cpu tensor
    # shape (n_nodes, 3)

    # atom type should be cpu tensor
    # shape (n_nodes,) (i.e. one hot converted to integer values)

    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    # ax.xaxis.pane.set_edgecolor('#D0D0D0')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == 'black':
        ax.w_xaxis.line.set_color("black")
    else:
        ax.w_xaxis.line.set_color("white")

    plot_molecule(ax, positions, atom_type, alpha, spheres_3d,
                  hex_bg_color, dataset_info)

    if 'qm9' in dataset_info['name']:
        max_value = positions.abs().max().item()

        # axis_lim = 3.2
        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    elif dataset_info['name'] == 'geom':
        max_value = positions.abs().max().item()

        # axis_lim = 3.2
        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    else:
        raise ValueError(dataset_info['name'])

    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        # plt.show()
        pass
    # plt.close()

# ------------------- end from visualize ------------------------

from training.structure import StructuredDataBatch


class QM9Dataset(StructuredDatasetBase):
    name="qm9"
    is_image=[0, 0, 0, 0, 0, 0, 0, 0, 0]
    is_onehot = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    names = ["pos", "atom_type", "charges", "alpha", "homo", "lumo", "gap", "mu", "Cv"]

    def __init__(self, random_rotation, subset, shuffle_node_ordering,
                 condition_on_alpha, only_second_half, pos_norm, atom_type_norm, charge_norm,
                 train_or_valid):
        print('------- making QM9 dataset --------')

        self.only_second_half = only_second_half

        cfg = get_cfg()
        if self.only_second_half:
            cfg.dataset = "qm9_second_half"
        self.dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        self.graphical_structure = QM9GraphicalStructure(
            max_dim=self.dataset_info['max_n_nodes']
        )

        self.shuffle_node_ordering = shuffle_node_ordering

        self.condition_on_alpha = condition_on_alpha
        cfg.datadir = './'

        args = init_argparse('qm9')
        args, datasets, num_species, charge_scale = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h)
        self.N = datasets['train'].num_pts

        # print(datasets['train'].data.keys())
        # print(datasets['train'].stats)
        self.subset = subset

        if subset > -1:
            self.indeces = torch.randperm(self.N)[0:subset]
            self.N = subset

        print('QM9 Dataset Length', self.N)

        self.num_atom_types = 5

        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                     'lumo': 27.2114}

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        # datasets['train'].data['positions'].shape (100000, 29, 3)
        # datasets['train'].data['charges'].shape (100000, 29)
        # datasets['train'].data['one_hot'].shape (100000, 29, 5)

        self.datasets = datasets

        # Construct PyTorch dataloaders from datasets
        # preprocess = PreprocessQM9(load_charges=args.include_charges)
        self.dataloaders = {'train': DataLoader(datasets['train'], batch_size=32)}


        self.random_rotation = random_rotation

        # self.norm_values = [1, 4, 10]
        self.norm_values = [pos_norm, atom_type_norm, charge_norm]

        self.condition_st_batch = None # for use in sampling

    def __getitem__(self, index, will_augment=False):
        assert not will_augment

        if self.subset > -1:
            index = self.indeces[index]

        positions = self.datasets['train'].data['positions'][index, ...]
        assert positions.shape == (self.graphical_structure.max_problem_dim, 3)
        positions = positions / self.norm_values[0]

        if self.random_rotation:
            positions = random_rotation(positions.view(1, -1, 3))[0, ...]

        atom_types = self.datasets['train'].data['one_hot'][index, ...]
        assert atom_types.shape == (self.graphical_structure.max_problem_dim, self.num_atom_types)
        atom_types = atom_types.float() / self.norm_values[1]

        charges = self.datasets['train'].data['charges'][index, ...]
        charges = charges.float() / self.norm_values[2]
        # charges is a max_dim length tensor with an integer representing each atom type.
        # it has zeros after that for non-atoms
        # so we can get the dimensionality of this datapoint from the charges
        problem_dim = torch.sum(charges > 0)
        
        if self.condition_on_alpha:
            alpha = (self.datasets['train'].data['alpha'][index, ...] - self.datasets['train'].stats['alpha'][0]) / self.datasets['train'].stats['alpha'][1]
            alpha = alpha.view(1)
        else:
            alpha = torch.zeros((1,))
        # homo = (self.datasets['train'].data['homo'][index, ...] - self.datasets['train'].stats['homo'][0]) / self.datasets['train'].stats['homo'][1]
        # lumo = (self.datasets['train'].data['lumo'][index, ...] - self.datasets['train'].stats['lumo'][0]) / self.datasets['train'].stats['lumo'][1]
        # gap = (self.datasets['train'].data['gap'][index, ...] - self.datasets['train'].stats['gap'][0]) / self.datasets['train'].stats['gap'][1]
        # mu = (self.datasets['train'].data['mu'][index, ...] - self.datasets['train'].stats['mu'][0]) / self.datasets['train'].stats['mu'][1]
        # Cv = (self.datasets['train'].data['Cv'][index, ...] - self.datasets['train'].stats['Cv'][0]) / self.datasets['train'].stats['Cv'][1]
        homo = torch.zeros((1,))
        lumo = torch.zeros((1,))
        gap = torch.zeros((1,))
        mu = torch.zeros((1,))
        Cv = torch.zeros((1,))
            

        if self.shuffle_node_ordering:
            perm = torch.randperm(problem_dim)
            positions[:problem_dim, :] = positions[perm, :]
            atom_types[:problem_dim, :] = atom_types[perm, :]
            charges[:problem_dim] = charges[perm]

        return problem_dim, positions, atom_types, charges, alpha, homo, lumo, \
               gap, mu, Cv

    def __len__(self):
        return self.N

    # def log_batch(self, data, gt_dims=None, wandb_log=True):
    def log_batch(self, in_st_batch, out_st_batch, wandb_log=True):

        molecules = {
            'x': out_st_batch.tuple_batch[0],
            'one_hot': out_st_batch.tuple_batch[1]
        }
        device = out_st_batch.get_device()
        batch_size = out_st_batch.tuple_batch[0].shape[0]
        max_n_nodes = out_st_batch.tuple_batch[0].shape[1]
        dims = out_st_batch.get_dims()
        node_mask = torch.zeros(batch_size, max_n_nodes)
        for i in range(batch_size):
            node_mask[i, 0:dims[i]] = 1
        node_mask = node_mask.unsqueeze(2).to(device)
        
        molecules['node_mask'] = node_mask
        
        # ---------------- From train_test.py/analyze_and_save ----------------
        validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules,
            self.dataset_info, dataloaders=self.dataloaders)

        combined_dict = {}
        for key in validity_dict:
            combined_dict[key] = validity_dict[key]
        if rdkit_tuple is not None:
            combined_dict['Validity'] = rdkit_tuple[0][0]
            combined_dict['Uniqueness'] = rdkit_tuple[0][1]
            combined_dict['Novelty'] = rdkit_tuple[0][2]

        if wandb_log:
            wandb.log(combined_dict)
            # if rdkit_tuple is not None:
            #     wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
        # ---------------------------------------------------------------------

        return combined_dict

    def condition_state(self, state_st_batch, condition_type, condition_sweep_idx=0, condition_sweep_path=None):
        """
            condition_type is a string for the type of conditioning

            condition_st_batch should have the correct number of dimensions

            if condition_type is 'sweep' then we use condition_sweep_idx and condition_sweep_path
            we load the atom numbers from condition sweep path and then pick the idx
        """
        batch = state_st_batch.B
        device = state_st_batch.get_device()
        num_dims = state_st_batch.get_dims()

        if (self.condition_st_batch is None) or (self.condition_st_batch.B != batch):
            if condition_type == 'sweep':
                assert condition_sweep_path is not None
                condition_data = np.load(condition_sweep_path)
                assert condition_data.shape == (10, 6) # first 4 numbers are C N O F next two can ignore
                condition_row = torch.from_numpy(condition_data[condition_sweep_idx, 0:4]).int()

                total_cond_dim = condition_row.sum()
                one_hot_cond = torch.zeros(total_cond_dim, 5)
                for i in range(4):
                    one_hot_cond[condition_row[0:i].sum():condition_row[0:i+1].sum(), i+1] = 1 # i+1 because we're not doing H


                self.condition_st_batch = StructuredDataBatch.create_copy(state_st_batch)    
                n_nodes = self.condition_st_batch.gs.max_problem_dim
                condition_flat_lats = torch.zeros_like(self.condition_st_batch.get_flat_lats()) # (B, 261)
                condition_flat_lats[:, 3*n_nodes:3*n_nodes + 5 * total_cond_dim] = one_hot_cond.float().flatten() / self.norm_values[1]
                self.condition_st_batch.set_flat_lats(condition_flat_lats)
                self.condition_st_batch.set_dims(total_cond_dim * torch.ones((batch,)))
                self.condition_st_batch.delete_dims(new_dims=total_cond_dim * torch.ones((batch,)))
                self.condition_mask = torch.zeros_like(self.condition_st_batch.get_flat_lats()) # (B, 261)
                self.condition_mask[:, 3*n_nodes:3*n_nodes + 5 * total_cond_dim] = torch.ones((5*total_cond_dim,)).to(device)
            else:
                raise ValueError('Condition type not recognized: ', condition_type)

        return self.condition_st_batch, self.condition_mask

        # if condition_method == 'set_xt':
        #     condition_mean, condition_std = loss.noise_schedule.get_p0t_stats(self.condition_st_batch, ts)
        #     condition_xt = condition_mean + rnd.randn_like(condition_std) * condition_std

        #     xt = state_st_batch.get_flat_lats()
        #     xt = self.condition_mask * condition_xt + (1-self.condition_mask) * xt
        #     state_st_batch.set_flat_lats(xt)
        #     state_st_batch.delete_dims(new_dims=num_dims)
        #     state_st_batch.gs.adjust_st_batch(state_st_batch)
        # else:
        #     raise ValueError('Condition method not recognized: ', condition_method)


class QM9GraphicalStructure(GraphicalStructureBase):
    def __init__(self, max_dim):
        self.max_problem_dim = max_dim
        cfg = get_cfg()
        self.dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)
        histogram = self.dataset_info['n_nodes']
        self.nodes_dist = DistributionNodes(histogram)

    def shapes_without_onehot(self):
        k = self.max_problem_dim
        return [torch.Size([k, 3]), torch.Size([k]), torch.Size([k]), \
                torch.Size([1]), torch.Size([1]), torch.Size([1]), \
                torch.Size([1]), torch.Size([1]), torch.Size([1])
        ]

    def shapes_with_onehot(self):
        k = self.max_problem_dim
        return [torch.Size([k, 3]), torch.Size([k, 5]), torch.Size([k]),
                torch.Size([1]), torch.Size([1]), torch.Size([1]), \
                torch.Size([1]), torch.Size([1]), torch.Size([1])
        ]

    def remove_problem_dims(self, data, new_dims):
        pos, atom_type, charge, alpha, homo, lumo, gap, mu, Cv = data


        B = pos.shape[0]
        assert pos.shape == (B, *self.shapes_with_onehot()[0])
        assert atom_type.shape == (B, *self.shapes_with_onehot()[1])
        assert charge.shape == (B, *self.shapes_with_onehot()[2])

        # for b_idx in range(B):
        #     pos[b_idx, new_dims[b_idx]:, :] = 0.0
        #     cats[b_idx, new_dims[b_idx]:, :] = 0.0
        #     ints[b_idx, new_dims[b_idx]:] = 0.0

        # pos, cats, ints = data
        device = pos.device
        new_dims_dev = new_dims.to(device)

        pos_mask = torch.arange(pos.shape[1], device=device).view(1, -1, 1).repeat(pos.shape[0], 1, pos.shape[2])
        pos_mask = (pos_mask < new_dims_dev.view(-1, 1, 1))
        pos = pos * pos_mask

        atom_type_mask = torch.arange(atom_type.shape[1], device=device).view(1, -1, 1).repeat(atom_type.shape[0], 1, atom_type.shape[2])
        atom_type_mask = (atom_type_mask < new_dims_dev.view(-1, 1, 1))
        atom_type = atom_type * atom_type_mask

        charge_mask = torch.arange(charge.shape[1], device=device).view(1, -1).repeat(charge.shape[0], 1)
        charge_mask = (charge_mask < new_dims_dev.view(-1, 1))
        charge = charge * charge_mask

        return pos, atom_type, charge, alpha, homo, lumo, gap, mu, Cv

    def adjust_st_batch(self, st_batch):
        device = st_batch.get_device()
        n_nodes = st_batch.gs.max_problem_dim
        B = st_batch.B
        dims = st_batch.get_dims()

        nan_batches = torch.isnan(st_batch.get_flat_lats()).any(dim=1).long().view(B,1)
        if nan_batches.sum() > 0:
            print('nan batches: ', nan_batches.sum())
        st_batch.set_flat_lats(torch.nan_to_num(st_batch.get_flat_lats()))


        x0_pos = st_batch.tuple_batch[0]
        assert x0_pos.shape == (B, n_nodes, 3)


        atom_mask = torch.arange(n_nodes).view(1, -1) < dims.view(-1, 1) # (B, n_nodes)
        assert atom_mask.shape == (B, n_nodes)
        atom_mask = atom_mask.long().to(device)
        node_mask = atom_mask.unsqueeze(2)
        assert node_mask.shape == (B, n_nodes, 1)

        # if any dims are 0 then set the node mask to all 1's. otherwise you get nans
        # all these results will be binned later anyway
        node_mask[dims==0, ...] = torch.ones((B, n_nodes, 1), device=device)[dims==0, ...].long()

        masked_max_abs_value = (x0_pos * (1 - node_mask)).abs().sum().item()
        assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
        N = node_mask.sum(1, keepdims=True)

        mean = torch.sum(x0_pos, dim=1, keepdim=True) / N
        assert mean.shape == (B, 1, 3)
        x0_pos = x0_pos - mean * node_mask

        assert x0_pos.shape == (B, n_nodes, 3)
        st_batch.set_flat_lats(torch.cat([
            x0_pos.flatten(start_dim=1),
            st_batch.tuple_batch[1].flatten(start_dim=1),
            st_batch.tuple_batch[2].flatten(start_dim=1)
        ], dim=1))
        return mean

    def get_auto_target(self, st_batch, adjust_val):
        B = st_batch.B
        device = st_batch.get_device()
        n_nodes = st_batch.gs.max_problem_dim
        assert adjust_val.shape == (B, 1, 3) # CoM of delxt
        delxt_CoM = adjust_val

        xt_pos = st_batch.tuple_batch[0]
        assert xt_pos.shape == (B, n_nodes, 3)
        atom_mask = torch.arange(n_nodes).view(1, -1) < st_batch.get_dims().view(-1, 1) # (B, n_nodes)
        assert atom_mask.shape == (B, n_nodes)
        atom_mask = atom_mask.long().to(device)
        node_mask = atom_mask.unsqueeze(2)
        assert node_mask.shape == (B, n_nodes, 1)

        xt_pos_from_y = (xt_pos - delxt_CoM) * node_mask

        assert xt_pos_from_y.shape == (B, n_nodes, 3)

        auto_target = torch.cat([
            xt_pos_from_y.flatten(start_dim=1),
            st_batch.tuple_batch[1].flatten(start_dim=1),
            st_batch.tuple_batch[2].flatten(start_dim=1)
        ], dim=1)
        assert auto_target.shape == (B, n_nodes * (3+5+1))

        return auto_target

    def get_nearest_atom(self, st_batch, delxt_st_batch):
        # assumes we are doing final dim deletion
        B = st_batch.B
        device = st_batch.get_device()

        x_full = st_batch.tuple_batch[0] # (B, n_nodes, 3)
        full_dims = st_batch.get_dims() # (B,)
        x_del = delxt_st_batch.tuple_batch[0] # (B, n_nodes, 3)

        # if full dim is 3 then x_full is [0, 1, 2] so missing atom is at idx 2

        missing_atom_pos = x_full[torch.arange(B, device=device).long(), (full_dims - 1).long(), :] # (B, 3)

        distances_to_missing = torch.sum((x_del - missing_atom_pos.unsqueeze(1)) ** 2, dim=2) # (B, n_nodes)

        atom_mask = torch.arange(st_batch.gs.max_problem_dim).view(1, -1) < delxt_st_batch.get_dims().view(-1, 1) # (B, n_nodes)
        atom_mask = atom_mask.to(device).long()

        distances_to_missing = atom_mask * distances_to_missing + (1-atom_mask) * 1e3

        nearest_atom = torch.argmin(distances_to_missing, dim=1) # (B,)

        return nearest_atom




qm9_datasets_to_kwargs = {
    QM9Dataset: set([
        ("random_rotation", "str2bool", "False"),
        ("subset", "int", -1),
        ("shuffle_node_ordering", "str2bool", "False"),
        ("condition_on_alpha", "str2bool", "False"),
        ("only_second_half", "str2bool", "False"),
        ("pos_norm", "float", 1.0),
        ("atom_type_norm", "float", 4.0),
        ("charge_norm", "float", 10.0),
        ("train_or_valid", "str", "train"), # this is ignored
    ]),
}
qm9_kwargs_gettable_from_dataset = {QM9Dataset: []}









# ----------------------- rdkit functions -----------------------


def compute_qm9_smiles(dataset_name, remove_h, dataloaders):
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print("\tConverting QM9 dataset to SMILES ...")

    class StaticArgs:
        def __init__(self, dataset, remove_h):
            self.dataset = dataset
            self.batch_size = 1
            self.num_workers = 1
            self.filter_n_atoms = None
            self.datadir = 'qm9/temp'
            self.remove_h = remove_h
            self.include_charges = True
    args_dataset = StaticArgs(dataset_name, remove_h)

    # dataloaders, charge_scale = dataset.retrieve_dataloaders(args_dataset)

    dataset_info = get_dataset_info(args_dataset.dataset, args_dataset.remove_h)
    n_types = 4 if remove_h else 5
    mols_smiles = []
    for i, data in enumerate(dataloaders['train']):
        positions = data['positions'][0].view(-1, 3).numpy()
        one_hot = data['one_hot'][0].view(-1, n_types).type(torch.float32)
        atom_type = torch.argmax(one_hot, dim=1).numpy()

        mol = build_molecule(torch.tensor(positions), torch.tensor(atom_type), dataset_info)
        mol = mol2smiles(mol)
        if mol is not None:
            mols_smiles.append(mol)
        if i % 1000 == 0:
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i)/len(dataloaders['train'])))
    return mols_smiles


def retrieve_qm9_smiles(dataset_info, dataloaders):
    dataset_name = dataset_info['name']
    if dataset_info['with_h']:
        pickle_name = dataset_name
    else:
        pickle_name = dataset_name + '_noH'

    file_name = 'qm9/temp/%s_smiles.pickle' % pickle_name
    try:
        with open(file_name, 'rb') as f:
            qm9_smiles = pickle.load(f)
        return qm9_smiles
    except OSError:
        try:
            os.makedirs('qm9/temp')
        except:
            pass
        qm9_smiles = compute_qm9_smiles(dataset_name, remove_h=not dataset_info['with_h'], dataloaders=dataloaders)
        with open(file_name, 'wb') as f:
            pickle.dump(qm9_smiles, f)
        return qm9_smiles


#### New implementation ####

# bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
#                 Chem.rdchem.BondType.AROMATIC]


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, dataloaders, dataset_smiles_list=None):
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for qm9 currently.
        if dataset_smiles_list is None and 'qm9' in dataset_info['name']:
            self.dataset_smiles_list = retrieve_qm9_smiles(
                self.dataset_info, dataloaders)

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []

        for graph in generated:
            mol = build_molecule(*graph, self.dataset_info)
            smiles = mol2smiles(mol)
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                valid.append(smiles)

        return valid, len(valid) / len(generated)

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity = self.compute_validity(generated)
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
        return [validity, uniqueness, novelty], unique


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(positions, atom_types, dataset_info):
    atom_decoder = dataset_info["atom_decoder"]
    X, A, E = build_xae_molecule(positions, atom_types, dataset_info)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
    return mol


def build_xae_molecule(positions, atom_types, dataset_info):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    atom_decoder = dataset_info['atom_decoder']
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_first_half':
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif dataset_info['name'] == 'geom':
                raise NotImplementedError
                # order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j], limit_bonds_to_one=True)
            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E












