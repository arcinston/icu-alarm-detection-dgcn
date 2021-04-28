# ICU_Alarm_Identification_Net

### Environment Configuration
If you want to run the project, first you need to install manually:
- pytorch=1.4
- numpy
- pandas
- wfdb
- prettytable
- scikit-learn
- matplotlib
- tensorwatch
- tensorboardX
- torchsummary
- joblib
- ...(You need to supplement the unmentioned content according to the operation prompt)

### You Should Know
The results in the current environment only means that the code can run normally. If you need the results which show in the paper, please use the trained model under the folder 'code/submit_entry/checkpoints/'. If you need to train models by yourself, you need to adjust 'code/options/default_opt.py's hyper-parameters.

### Train Model
Just modify the 'main_physionet2015.py' file as needed and execute it in the configured operating environment:

```python
# conda activate YOUR_ENVIRONMENT
python main_physionet2015.py
```

The trained model will be saved in the checkpoints folder, and the training log will be kept in the LOG folder.


### Evaluate Model
If you need to test in the sandbox provided by physionet2015, you need to manually install the operating environment in the sandbox.

Our ‘submit_entry’ folder provides some trained models and sandbox environment configuration scripts. Because the 'mkl-2020.1-217.conda' file is too large, we have moved it from the '/submit_entry/conda_pkgs/' folder to the '/data' folder. You need to manually download the file to the '/submit_entry/conda_pkgs/' folder, package the file in the directory into an 'entry.zip' file, and submit it to the physiont2015 sandbox environment to get the test result.