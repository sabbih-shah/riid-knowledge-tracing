# riid-knowledge-tracing
A Saintplus Transformer implementation for the Riiid Answer Correctness Prediction challenge. 


### Training:
To train the model from scratch download the dataset from the below link:

https://www.kaggle.com/c/riiid-test-answer-prediction/data

After downloading the dataset modify the *config.py* file with desired parameters. A pretrained embedding *riid_256_embedding_400.npz* is provided, which was trained by building directional graph and link between tags (used as requied skills) and metadata from *questions.csv*.  

