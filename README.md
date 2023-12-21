# Fine-tuning a pre-trained LLM

## License
This template is licensed under Apache 2.0 and contains the following components: 
* finBERT [Apache 2.0](https://github.com/ProsusAI/finBERT/blob/master/LICENSE)
* pytorch [Caffe 2](https://github.com/pytorch/pytorch/blob/main/LICENSE)
* NVIDIA [EULA](https://docs.nvidia.com/cuda/eula/index.html#license-grant)
* Transformer [Apache 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE)
* pandas [BSD 3](https://github.com/pandas-dev/pandas/blob/main/LICENSE)
* scikit-Learn [BSD 3](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)
* numpy [License](https://numpy.org/doc/stable/license.html)

## About this project
In this project we demonstrate the use of a pre-trained Large Language Model (LLM) in Domino and the process of fine-tuning the model for a specific task.

Fine-tuning a pre-trained LLM is a commonly used technique for solving NLP problems with machine learning. This is a typical transfer learning task where the final model is realised through a number of training phases:

1. The process typically begins with a pre-trained model, which is not task specific. This model is trained on a large corpora of unlabelled data (e.g. Wikipedia) using masked-language modelling loss. Bidirectional Encoder Representations from Transformers (BERT) is an example for such a model [1].

2. The model undergoes a process of domain specific adaptive fine-tunning, which produces a new model with narrower focus. This new model is better prepared to address domain-specific challenges as it is now closer to the expected distribution of the target data. An example for such a model is FinBERT [2].

3. The final stage of the model building pipeline is the behavioural fine-tuning. This is the step where the model is further fine-tuned using a specific supervised learning dataset. The resulting model is now well equipped to solve one specific problem.

The advantage of using the process outlined above is that pre-training is typically computationally expensive and requires the processing of large volumes of data. Behavioural fine-tuning, on the other hand, is relatively cheap and quick to accomplish, especially if distributed and GPU-accelerated compute can be employed in the process.

In this demo project we use the [Sentiment Analysis for Financial News dataset](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) [3], which provides 4,837 samples of sentiments for financial news headlines from the perspective of a retail investor. This dataset is used in conjuction with FinBERT, which we fine-tune for the purposes of performing sentiment analysis.

The assets available in this project are:

* **finetune.ipynb** - A notebook, illustrating the process of getting FinBERT from [Huggingface 🤗](https://huggingface.co/ProsusAI/finbert) into Domino, and using GPU-accelerated backend for the purposes of fine-tuning it with the Sentiment Analysis for Financial News dataset
* **finetune.py** - A script, which performs the same fine-tuning process but can be scheduled and executed as a Domino job. The script also accepts two command line arguments:
    * *lr* - learning rate for the fine-tunning process
    * *epochs* - number of training epochs
* **all-data.csv** - A CSV file containing the Sentiment Analysis for Financial News dataset
* **score.py** - A scoring function, which is used to deploy the fine-tuned model as a [Domino API](https://docs.dominodatalab.com/en/latest/user_guide/8dbc91/host-models-as-rest-apis/)
* **streamlit_app.py** - Code for the streamlit app. Please note that the app uses a model API, **please change the model URL and token in this file after you have started the model API**
* **app.sh** - Launch instructions for the accompanying Streamlit app

# Model API calls

Please ensure that all artifacts from the workspace after the fine tuning have been synced before starting the model API

The **score.py** provides a scoring function with the following signature: `predict_sentiment(sentence)`. To test it, you could use the following JSON payload:

```
{
  "data": {
    "sentence": "there is a shortage of capital, and we need extra financing"
  }
}
```

# Set up instructions

This project requires the following [compute environments](https://docs.dominodatalab.com/en/latest/user_guide/f51038/environments/) to be present. Please ensure the "Automatically make compatible with Domino" checkbox is selected while creating the environment.

## Environment Requirements

`quay.io/domino/pre-release-environments:project-hub-gpu.main.latest`

**Pluggable Workspace Tools** 
```
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: [ "/opt/domino/workspaces/jupyterlab/start" ]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
vscode:
 title: "vscode"
 iconUrl: "/assets/images/workspace-logos/vscode.svg"
 start: [ "/opt/domino/workspaces/vscode/start" ]
 httpProxy:
    port: 8888
    requireSubdomain: false
```

Please change the value in `start` according to your Domino version.

**Important !!** The version of PyTorch included in the above image is not compatible with GPUs with the Ampere architecture. Please check the PyTorch compatibility with your GPU before running this code. 

The following environment variables should also be set at the project level:

```
DISABLE_MLFLOW_INTEGRATION=TRUE	
TF_ENABLE_ONEDNN_OPTS=0
```

## Hardware Requirements
You also need to make sure that the hardware tier running the notebook or the fine-tuning script has sufficient resources. An *nvidia-low-g4dn-xlarge* hardware tier is recommended, as it provides GPU-acceleration that the fine-tunning can take advantage of.

# References

[1] Devlin, Jacob; Chang, Ming-Wei; Lee, Kenton; Toutanova, Kristina, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, 11 October 2018, [arXiv:1810.04805v2](https://arxiv.org/abs/1810.04805)

[2] Dogu Araci, FinBERT: Financial Sentiment Analysis with Pre-trained Language Models, 2019, [arXiv:1908.10063](http://arxiv.org/abs/1908.10063)

[3] Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, 65(4), 782-796.

