# FGVC
Fine-Grained Visual Classification

## Seed Image Classification
The dataset used for this project is available [here](https://www.nature.com/articles/s41597-024-03176-5#ref-CR30).

![Predicted Samples.](/assets/prediction.png)



Models and Accuracy
In this project, multiple models were tried and their accuracies are reported below:

Resnet18: Accuracy - 88.9%

Resnet34: Accuracy - 92.11%

CMAL: Accuracy - 94.89%

### CMAL - Cross-layer Mutual Attention Learning

This model was implemented based on the [paper](https://www.sciencedirect.com/science/article/pii/S0031320323002509) 

This model proposes a cross-layer mutual attention learning network (CMAL-Net) to solve the FGVC problems. Specifically, this work views the shallow to deep layers of CNNs as “experts” knowledgeable about different perspectives. We let each expert give a category prediction and an attention region indicating the found clues. Attention regions are treated as information carriers among experts, bringing three benefits: 
* helping the model focus on discriminative regions;
* providing more training data; 
* allowing experts to learn from each other to improve the overall performance. 
