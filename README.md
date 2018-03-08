# Seq2seq code in PyTorch
Building from [Ruotian Luo's code for captioning](https://github.com/ruotianluo/ImageCaptioning.pytorch)
AND [Sandeep Subramanian's seq2seq code](https://github.com/MaximumEntropy/Seq2Seq-PyTorch)

### Data preprocessing:
I use these steps from [Alexandre Bérard's code](https://github.com/eske/seq2seq)

    > config/WMT14/download.sh    # download WMT14 data into raw_data/WMT14
    > config/WMT14/prepare.sh     # preprocess the data, and copy the files to data/WMT14

Then run the following to save in h5 files:

    > python scripts/prepro_text.py 

### Training:

Training requires some directories for saving the model's snapshots, the tensorboard events 

    > mkdir -p save events

To train a model under the parameters defined in config.yaml

    > python nmt.py -c config.yaml 

Check **options/opts.py** for more about the options.

To evaluate a model:

    > python eval.py -c config


To submit jobs via OAR use either _train.sh_ or _select_train.sh_
