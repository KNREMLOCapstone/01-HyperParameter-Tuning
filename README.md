I have used the Lightning-Hydra-Template for Hyperparameter tuning

Results of optimizayion_results.yaml for resnet18, resnet34, vit_base_patch16_224

<u>1. Resnet18</u>
name: optuna
best_params:
  model.optimizer.lr: 0.059362191911672815
  model.optimizer._target_: torch.optim.SGD
best_value: 0.9606438875198364

<u>2. Resnet34</u>
name: optuna
best_params:
  model.optimizer.lr: 0.06248029596659127
  model.optimizer._target_: torch.optim.SGD
best_value: 0.9560036659240723

<u>3. vit_base_patch16_224
 name: optuna
best_params:
  model.optimizer.lr: 0.0476321232531695
  model.optimizer._target_: torch.optim.SGD
best_value: 0.9681627750396729
 
 ![image](https://user-images.githubusercontent.com/16095633/218397360-3770964d-6431-4b4c-b294-80cf9630b2a7.png)

 Since the vit_base model is consuming so much cuda memory, I went ahead with resnet18 for further steps. :(
  
  The hyper parameter tuning is based on the "test/best_acc". 
 
 I have logged in the metrics asked
    F1 Score
    Precision
    Recall
    Confusion Matrix (Can be Image in Tensorboard)
 
Tensorboard
https://tensorboard.dev/experiment/nKXg7Bu4QwWLc3mvp61IQA/

Confusion Matrix
![image](https://user-images.githubusercontent.com/16095633/218531491-cc615baf-5f58-43ed-8ce9-0536a764118c.png)


 
