# Homegrown LLM
A GPT model that is designed to run on frontend, backend, and database code.

## Getting Started
1. Open training_config.py. 
2. Set is_first_run=True. 
3. Set training_dataset_dir to a valid empty folder. This is where the model will be outputted. 
4. Set output_root to valid directories.

## How to interpret training output
Example log:

`84%|████████▍ | 320/380 [1:27:33<14:12, 14.21s/it] {'loss': 0.0602, 'grad_norm': 1.818, 'learning_rate': 4.97e-06, 'epoch': 4.21}`

* `84%` You're 84% through the entire training process (in terms of training steps).
* `320/380` You've completed 320 out of 380 total training steps.
* `[1:27:33<14:12]` 1:27:33 Total elapsed time. 14:12 Estimated time remaining.
* `14.21s/it` Each training step (iteration) is taking about 14.21 seconds.
* `loss: 0.0602` This is the training loss. Lower is better. It measures how far off the model's predictions are from the actual data. It’s already very low — your model is learning well.
* `grad_norm: 1.818` The gradient norm measures the magnitude of the gradients during backpropagation. High values may indicate instability; this is within a healthy range.
* `learning_rate: 4.97e-06` Current learning rate being used. Since your trainer is likely using linear decay, this rate is going down as training progresses.
* `epoch: 4.21` You are 21% into the 5th epoch. (Training is defined to run for 5 epochs in your config.)