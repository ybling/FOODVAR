# The implementation of FOODVAR in the "Consumption Variety in Food Recommendation" paper

This repository contains the implementations of FOODVAR model for incorporating variety into the food recommendation system.

## Requirements

- python == 3.7.4
- torch == 1.0.1
- pandas == 0.25.1
- numpy == 1.17.2
- pickle == 4.0
- tqdm


## Data

Because of NDA conditions, the real data cannot be posted. The small synthetic sample data that abids by NDA conditions are given in the `/data` folder.


### Data Format

See data format in the `data` folder which including the sample data files.

The Train/Validation/Test set are all `JSON` file.
- `userID` is the unique ID for the specific user
- `baskets` is the list(`A`) of lists(`B`). `A` contains all the items consumed by the user. `B` is formed by two parts: The last number is the variety change indicator (`0` refers to variety maintenace, `1` refers to variety increase and `2` refers to variety decrease); The other part is the items consumed for the same day. 
- `num_baskets` is the total number of days the user logged in.

This repository can be used in other recommendation datasets in two ways:
1. Modify your datasets into the same format of the sample files.
2. Modify the data preprocess code in `utils/data_helpers.py`.


## How to train
you can directly run the `train.py` and `Test.py` files to train and test the model:    
`python train.py`     
`python test.py`



## Citation
If you find this code useful in your research, please consider citing:       

`Nielsen, Daiva, Nathan Yang, Laurette Dubé, Bärbel Knäuper, Yabo Ling, and Jian-Yun Nie (2020), "Consumption Variety in Food Recommendation”`     

The full paper is available for download at: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3707438

