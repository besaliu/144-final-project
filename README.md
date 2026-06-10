# 144-final-project


CSE144 Final Project
COntributions by: 
Ben Liu (CruzID: 1932243)
Milan Moslehi (CruzID: 1977483)
Aaron Yam (CruzID: 2105044)


To understand our project, please refer to the CSE144 Report.pdf and for our weights you can go to the folder labeled checkpoints->best_model.pt or reference the google link below
https://drive.google.com/file/d/1z9NXb1m_oRLn3R8NAWz_Q6SRgEus2f-p/view?usp=sharing

Check out the final Report and our Presentation linked in the files of the repo.


How To Run Our Repo:

1. Random seeds and determinism settings.
SEED = 43
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
Validation split uses random.Random(SEED) for per-class shuffling. DataLoader uses num_workers=0 for deterministic behavior on macOS.
2. Package versions / environment setup steps.
Clone the repository and place the dataset under ucsc-cse-144-spring-2026-final-project/.
Install dependencies:
pip install torch torchvision numpy pandas pillow tqdm certifi
Verified environment: PyTorch 2.8.0, Python 3.x, MPS or CUDA device.
3. Exact commands to train and to generate submission.csv.
Open and run all cells in train.ipynb, or equivalently:
jupyter notebook train.ipynb
# Run all cells sequentially
The notebook will:
Train Phase 1 (5 epochs, frozen backbone)
Train Phase 2 (up to 45 epochs with early stopping)
Save best model to checkpoints/best_model.pt
Load best checkpoint, run TTA inference, and write submission.csv
https://drive.google.com/file/d/1z9NXb1m_oRLn3R8NAWz_Q6SRgEus2f-p/view?usp=sharing 
Link to weights in Google Drive, if it does not work you can view in the github repo from checkpoints->best_model.pt

For further reference ins cell block 2 in train.ipynb:
SEED = 43

IMG_SIZE = 300
BATCH_SIZE = 12
NUM_CLASSES = 100
NUM_WORKERS = 0

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 60

PHASE1_LR = 8e-4
PHASE2_LR = 6e-5

WEIGHT_DECAY = 7e-4
LABEL_SMOOTH = 0.03
MIXUP_ALPHA = 0.0

PATIENCE = 14


How to Run Inference (Testing & Submissions):
If you want to evaluate our final model or generate a Kaggle submission file without retraining, you can download our pre-trained weights and run the inference pipeline.
1. Download Pre-trained WeightsDownload our best_model.pt file from Google Drive and place it in the root directory of this repository:Download Trained Model Weights (Google Drive Link)
2. Generate PredictionsRun the inference script to load the test dataset, preprocess/resize the images to $224\times224$, apply the downloaded model weights, and generate predictions mapping strictly to the integer labels (0 for class "0", 1 for class "1", etc.).
Bashpython inference.py --data_dir ./data/test --weights ./best_model.pt --output ./submission.csv
This will output a correctly formatted submission.csv file containing the {ID, Label} columns, ready for Kaggle submission evaluation.



From Our Report:

Brief summary of your approach and main result.
We use EfficientNet V2-S pretrained on ImageNet, replace the default classifier with a deeper head (512-d hidden layer, SiLU activation, dropout), and train in two phases: (1) frozen backbone, classifier-only warmup for 5 epochs; (2) full fine-tuning for up to 45 epochs with early stopping. We apply moderate data augmentation, label smoothing, discriminative learning rates, and 4-view test-time augmentation (TTA) at inference. On our stratified validation set (1 image per class), the best checkpoint achieves 84.0% validation accuracy at epoch 23 (global epoch 28). Final predictions are generated with TTA and saved to submission.csv.

<img width="1269" height="1180" alt="image" src="https://github.com/user-attachments/assets/d50dca6b-f223-46be-baf8-20368f6c15e8" />
