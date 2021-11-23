# CasDO
ICDE 2022
Paper: Probabilistic Modelling of Irregularly Sampled Information Cascades with Evolution Uncertainty
### Requirements
The code was tested with `python 3.7`, `pyorch-GPU 1.10`, `cudatookkit 10.2`, and `cudnn 7.6.5`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name CasDO python=3.7 cudatoolkit=10.2 cudnn=7.6.5

# activate environment
conda activate CasDO

# install other requirements
pip install -r requirements.txt
```

### Run the code
```shell
cd ./CasDO

# generate information cascades
python gene_cas.py --input=./dataset/twitter/

# generate cascade graph and global graph embeddings 
python gene_emb.py --input=./dataset/twitter/

# run CasDO model
python run_model.py --input=./dataset/twitter/
```
More running options are described in the codes, e.g., `--input=./dataset/twitter/`

## Datasets

Datasets download link: [Google Drive](https://drive.google.com/file/d/1o4KAZs19fl4Qa5LUtdnmNy57gHa15AF-/view?usp=sharing) or [Baidu Drive (password: `1msd`)](https://pan.baidu.com/s/1tWcEefxoRHj002F0s9BCTQ).

The datasets we used in the paper are come from:

- [Twitter](http://carl.cs.indiana.edu/data/#virality2013) (Weng *et al.*, [Virality Prediction and Community Structure in Social Network](https://www.nature.com/articles/srep02522), Scientific Report, 2013).
- [Weibo](https://github.com/CaoQi92/DeepHawkes) (Cao *et al.*, [DeepHawkes: Bridging the Gap between 
Prediction and Understanding of Information Cascades](https://dl.acm.org/doi/10.1145/3132847.3132973), CIKM, 2017). You can also download Weibo dataset [here](https://drive.google.com/file/d/1fgkLeFRYQDQOKPujsmn61sGbJt6PaERF/view?usp=sharing) in Google Drive.  
- [APS](https://journals.aps.org/datasets) (Released by *American Physical Society*, obtained at Jan 17, 2019). 

## Contact
For any questions please open an issue or drop an email to: `zhangtao980107@outlook.com`
