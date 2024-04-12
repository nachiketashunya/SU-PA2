"""
Data preparation.

Download: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
"""

import csv
import glob
import logging
import os
import random
import shutil
import sys  # noqa F401

import numpy as np
import torch
import torchaudio
from tqdm.contrib import tqdm
import wandb 

from speechbrain.dataio.dataio import load_pkl, save_pkl

logger = logging.getLogger(__name__)
OPT_FILE = "opt_voxceleb_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"
ENROL_CSV = "enrol.csv"
SAMPLERATE = 16000


DEV_WAV = "vox1_dev_wav.zip"
TEST_WAV = "vox1_test_wav.zip"
META = "meta"


def prepare_voxceleb(
    data_folder,
    save_folder,
    verification_pairs_file,
    splits=["train", "dev", "test"],
    split_ratio=[90, 10],
    seg_dur=3.0,
    amp_th=5e-04,
    source=None,
    split_speaker=False,
    random_segment=False,
    skip_prep=False,
):
    """
    Prepares the csv files for the Voxceleb1 or Voxceleb2 datasets.
    Please follow the instructions in the README.md file for
    preparing Voxceleb2.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original VoxCeleb dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    verification_pairs_file : str
        txt file containing the verification split.
    splits : list
        List of splits to prepare from ['train', 'dev']
    split_ratio : list
        List if int for train and validation splits
    seg_dur : int
        Segment duration of a chunk in seconds (e.g., 3.0 seconds).
    amp_th : float
        removes segments whose average amplitude is below the
        given threshold.
    source : str
        Path to the folder where the VoxCeleb dataset source is stored.
    split_speaker : bool
        Speaker-wise split
    random_segment : bool
        Train random segments
    skip_prep : bool
        If True, skip preparation.

    Returns
    -------
    None

    Example
    -------
    >>> from recipes.VoxCeleb.voxceleb1_prepare import prepare_voxceleb
    >>> data_folder = 'data/VoxCeleb1/'
    >>> save_folder = 'VoxData/'
    >>> splits = ['train', 'dev']
    >>> split_ratio = [90, 10]
    >>> prepare_voxceleb(data_folder, save_folder, splits, split_ratio)
    """

    if skip_prep:
        return
    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "split_ratio": split_ratio,
        "save_folder": save_folder,
        "seg_dur": seg_dur,
        "split_speaker": split_speaker,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting output files
    save_opt = os.path.join(save_folder, OPT_FILE)
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_dev = os.path.join(save_folder, DEV_CSV)

    # Create the data folder contains VoxCeleb1 test data from the source
    if source is not None:
        if not os.path.exists(os.path.join(data_folder, "wav", "id10270")):
            logger.info(f"Extracting {source}/{TEST_WAV} to {data_folder}")
            shutil.unpack_archive(os.path.join(source, TEST_WAV), data_folder)
        if not os.path.exists(os.path.join(data_folder, "meta")):
            logger.info(f"Copying {source}/meta to {data_folder}")
            shutil.copytree(
                os.path.join(source, "meta"), os.path.join(data_folder, "meta")
            )

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Additional checks to make sure the data folder contains VoxCeleb data
    if "," in data_folder:
        data_folder = data_folder.replace(" ", "").split(",")
    else:
        data_folder = [data_folder]

    # _check_voxceleb1_folders(data_folder, splits)

    msg = "\tCreating csv file for the VoxCeleb Dataset.."
    logger.info(msg)

    # Split data into 90% train and 10% validation (verification split)
    wav_lst_train, wav_lst_dev = _get_utt_split_lists(
        data_folder, split_ratio, verification_pairs_file, split_speaker
    )

    # Creating csv file for training data
    if "train" in splits:
        prepare_csv(
            seg_dur, wav_lst_train, save_csv_train, random_segment, amp_th
        )

    if "dev" in splits:
        prepare_csv(seg_dur, wav_lst_dev, save_csv_dev, random_segment, amp_th)

    # For PLDA verification
    if "test" in splits:
        prepare_csv_enrol_test(
            data_folder, save_folder, verification_pairs_file
        )

    # Saving options (useful to skip this phase when already done)
    save_pkl(conf, save_opt)


def skip(splits, save_folder, conf):
    """
    Detects if the voxceleb data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    splits : list
    save_folder : str
    conf : str

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking csv files
    skip = True

    split_files = {
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
        "test": TEST_CSV,
        "enrol": ENROL_CSV,
    }
    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split_files[split])):
            skip = False
    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


def _check_voxceleb_folders(data_folders, splits):
    """
    Check if the data folder actually contains the Voxceleb1 dataset.

    If it does not, raise an error.

    Arguments
    ---------
    data_folders : list
        List of data folder paths to check
    splits : list
        List of splits, "train" and/or "test"

    Raises
    ------
    FileNotFoundError
    """
    for data_folder in data_folders:
        if "train" in splits:
            folder_vox1 = os.path.join(data_folder, "wav", "id10001")
            folder_vox2 = os.path.join(data_folder, "wav", "id00012")

            if not os.path.exists(folder_vox1) or not os.path.exists(
                folder_vox2
            ):
                err_msg = "the specified folder does not contain Voxceleb"
                raise FileNotFoundError(err_msg)

        if "test" in splits:
            folder = os.path.join(data_folder, "wav", "id10270")
            if not os.path.exists(folder):
                err_msg = (
                    "the folder %s does not exist (as it is expected in "
                    "the Voxceleb dataset)" % folder
                )
                raise FileNotFoundError(err_msg)

        folder = os.path.join(data_folder, "meta")
        if not os.path.exists(folder):
            err_msg = (
                "the folder %s does not exist (as it is expected in "
                "the Voxceleb dataset)" % folder
            )
            raise FileNotFoundError(err_msg)


# Used for verification split
def _get_utt_split_lists(
    data_folders, split_ratio, verification_pairs_file, split_speaker=False
):
    """
    Tot. number of speakers vox1= 1211.
    Tot. number of speakers vox2= 5994.
    Splits the audio file list into train and dev.
    This function automatically removes verification test files from the training and dev set (if any).
    """
    train_lst = []
    dev_lst = []

    print("Getting file list...")
    for data_folder in data_folders:
        test_lst = [
            line.rstrip("\n").split(" ")[1]
            for line in open(verification_pairs_file)
        ]
        test_lst = set(sorted(test_lst))

        test_spks = [snt.split("/")[0] for snt in test_lst]

        path = os.path.join(data_folder, "wav", "**", "*.wav")
        if split_speaker:
            # avoid test speakers for train and dev splits
            audio_files_dict = {}
            for f in glob.glob(path, recursive=True):
                spk_id = f.split("/wav/")[1].split("/")[0]
                if spk_id not in test_spks:
                    audio_files_dict.setdefault(spk_id, []).append(f)

            spk_id_list = list(audio_files_dict.keys())
            random.shuffle(spk_id_list)
            split = int(0.01 * split_ratio[0] * len(spk_id_list))
            for spk_id in spk_id_list[:split]:
                train_lst.extend(audio_files_dict[spk_id])

            for spk_id in spk_id_list[split:]:
                dev_lst.extend(audio_files_dict[spk_id])
        else:
            # avoid test speakers for train and dev splits
            audio_files_list = []
            for f in glob.glob(path, recursive=True):
                try:
                    spk_id = f.split("/wav/")[1].split("/")[0]
                except ValueError:
                    logger.info(f"Malformed path: {f}")
                    continue
                if spk_id not in test_spks:
                    audio_files_list.append(f)

            random.shuffle(audio_files_list)
            split = int(0.01 * split_ratio[0] * len(audio_files_list))
            train_snts = audio_files_list[:split]
            dev_snts = audio_files_list[split:]

            train_lst.extend(train_snts)
            dev_lst.extend(dev_snts)

    return train_lst, dev_lst


def _get_chunks(seg_dur, audio_id, audio_duration):
    """
    Returns list of chunks
    """
    num_chunks = int(audio_duration / seg_dur)  # all in milliseconds

    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]

    return chunk_lst


def prepare_csv(seg_dur, wav_lst, csv_file, random_segment=False, amp_th=0):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    seg_dur : int
        Segment duration of a chunk in seconds (e.g., 3.0 seconds).
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file
    random_segment: bool
        Read random segments
    amp_th: float
        Threshold on the average amplitude on the chunk.
        If under this threshold, the chunk is discarded.
    """

    msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    logger.info(msg)

    csv_output = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    # For assigning unique ID to each chunk
    my_sep = "--"
    entry = []
    # Processing all the wav files in the list
    for wav_file in tqdm(wav_lst, dynamic_ncols=True):
        # Getting sentence and speaker ids
        try:
            [spk_id, sess_id, utt_id] = wav_file.split("/")[-3:]
        except ValueError:
            logger.info(f"Malformed path: {wav_file}")
            continue
        audio_id = my_sep.join([spk_id, sess_id, utt_id.split(".")[0]])

        # Reading the signal (to retrieve duration in seconds)
        signal, fs = torchaudio.load(wav_file)
        signal = signal.squeeze(0)

        if random_segment:
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]

            # Composition of the csv_line
            csv_line = [
                audio_id,
                str(audio_duration),
                wav_file,
                start_sample,
                stop_sample,
                spk_id,
            ]
            entry.append(csv_line)
        else:
            audio_duration = signal.shape[0] / SAMPLERATE

            uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)
            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]
                start_sample = int(float(s) * SAMPLERATE)
                end_sample = int(float(e) * SAMPLERATE)

                #  Avoid chunks with very small energy
                mean_sig = torch.mean(np.abs(signal[start_sample:end_sample]))
                if mean_sig < amp_th:
                    continue

                # Composition of the csv_line
                csv_line = [
                    chunk,
                    str(audio_duration),
                    wav_file,
                    start_sample,
                    end_sample,
                    spk_id,
                ]
                entry.append(csv_line)

    csv_output = csv_output + entry

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    # Final prints
    msg = "\t%s successfully created!" % (csv_file)
    logger.info(msg)


def prepare_csv_enrol_test(data_folders, save_folder, verification_pairs_file):
    """
    Creates the csv file for test data (useful for verification)

    Arguments
    ---------
    data_folders : str
        Path of the data folders
    save_folder : str
        The directory where to store the csv files.
    verification_pairs_file : str
        Path to the file with verification pairs.
    """

    # msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    # logger.debug(msg)

    csv_output_head = [
        ["ID", "duration", "wav", "start", "stop", "spk_id"]
    ]  # noqa E231

    for data_folder in data_folders:
        test_lst_file = verification_pairs_file

        enrol_ids, test_ids = [], []

        # Get unique ids (enrol and test utterances)
        for line in open(test_lst_file):
            e_id = line.split(" ")[1].rstrip().split(".")[0].strip()
            t_id = line.split(" ")[2].rstrip().split(".")[0].strip()
            enrol_ids.append(e_id)
            test_ids.append(t_id)

        enrol_ids = list(np.unique(np.array(enrol_ids)))
        test_ids = list(np.unique(np.array(test_ids)))

        # Prepare enrol csv
        logger.info("preparing enrol csv")
        enrol_csv = []
        for id in enrol_ids:
            wav = data_folder + "/wav/" + id + ".wav"

            # Reading the signal (to retrieve duration in seconds)
            signal, fs = torchaudio.load(wav)
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]
            [spk_id, sess_id, utt_id] = wav.split("/")[-3:]

            csv_line = [
                id,
                audio_duration,
                wav,
                start_sample,
                stop_sample,
                spk_id,
            ]

            enrol_csv.append(csv_line)

        csv_output = csv_output_head + enrol_csv
        csv_file = os.path.join(save_folder, ENROL_CSV)

        # Writing the csv lines
        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_output:
                csv_writer.writerow(line)

        # Prepare test csv
        logger.info("preparing test csv")
        test_csv = []
        for id in test_ids:
            wav = data_folder + "/wav/" + id + ".wav"

            # Reading the signal (to retrieve duration in seconds)
            signal, fs = torchaudio.load(wav)
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]
            [spk_id, sess_id, utt_id] = wav.split("/")[-3:]

            csv_line = [
                id,
                audio_duration,
                wav,
                start_sample,
                stop_sample,
                spk_id,
            ]

            test_csv.append(csv_line)

        csv_output = csv_output_head + test_csv
        csv_file = os.path.join(save_folder, TEST_CSV)

        # Writing the csv lines
        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_output:
                csv_writer.writerow(line)

#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import os
import random
import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main


class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"""

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, lens = self.hparams.wav_augment(wavs, lens)

        # Feature extraction and normalization
        if (
            hasattr(self.hparams, "use_tacotron2_mel_spec")
            and self.hparams.use_tacotron2_mel_spec
        ):
            feats = self.hparams.compute_features(audio=wavs)
            feats = torch.transpose(feats, 1, 2)
        else:
            feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            spkid = self.hparams.wav_augment.replicate_labels(spkid)

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        wandb.log({
            'train_loss': loss
        })
        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="spk_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data, label_encoder


wandb.init(project="Speaker Verification", name="Fine Tune on KathBath")

 # This flag enables the inbuilt cudnn auto-tuner
torch.backends.cudnn.benchmark = True

# CLI:
hparams_file = "/content/train_ecapa_tdnn.yaml"
# Load hyperparameters file with command-line overrides
data_folder = "/content/valid"
overrides = f"data_folder: {data_folder}\noutput_folder: /content"


# Load hyperparameters file with command-line overrides
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

# Download verification list (to exclude verification sentences from train)
veri_file_path = os.path.join(
    hparams["save_folder"], os.path.basename(hparams["verification_file"])
)
download_file(hparams["verification_file"], veri_file_path)

run_on_main(
    prepare_voxceleb,
    kwargs={
        "data_folder": hparams["data_folder"],
        "save_folder": hparams["save_folder"],
        "verification_pairs_file": "/content/kb_valid_pairs (2).txt",
        "splits": ["train", "dev"],
        "split_ratio": hparams["split_ratio"],
        "seg_dur": hparams["sentence_len"],
        "skip_prep": hparams["skip_prep"],
    },
)
sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

# Dataset IO prep: creating Dataset objects and proper encodings for phones
train_data, valid_data, label_encoder = dataio_prep(hparams)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=hparams["output_folder"],
    hyperparams_to_save=hparams_file,
    overrides=overrides,
)

# Brain class initialization
speaker_brain = SpeakerBrain(
    modules=hparams["modules"],
    opt_class=hparams["opt_class"],
    hparams=hparams,
    run_opts={'device': 'cuda'},
    checkpointer=hparams["checkpointer"],
)

# Training
speaker_brain.fit(
    speaker_brain.hparams.epoch_counter,
    train_data,
    valid_data,
    train_loader_kwargs=hparams["dataloader_options"],
    valid_loader_kwargs=hparams["dataloader_options"],
)

wandb.finish()