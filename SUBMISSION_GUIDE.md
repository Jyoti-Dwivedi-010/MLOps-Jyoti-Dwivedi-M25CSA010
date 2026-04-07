# GitHub Submission Guide for Assignment 5

## Step 1: Download the Assignment Folder

### Option A: Using SCP (from your local terminal/PowerShell)
```powershell
# Run this from your LOCAL machine (not the server)
scp -r sujiv1@<server-ip>:/data/sujiv1/sujiv/jyoti/mlops/Assignment-5 .
```

### Option B: Using VS Code Remote
If you're using VS Code with Remote SSH, right-click the Assignment-5 folder and select "Download".

---

## Step 2: Initialize Git and Push to GitHub

Open PowerShell/Terminal in the downloaded Assignment-5 folder and run:

```powershell
# Navigate to the folder
cd Assignment-5

# Initialize git
git init

# Add remote (your repo)
git remote add origin https://github.com/Jyoti-Dwivedi-010/MLOps-Jyoti-Dwivedi-M25CSA010.git

# Create and switch to Assignment-5 branch
git checkout -b "Assignment-5"

# Add all files
git add .

# Commit
git commit -m "Assignment 5: ViT-S LoRA + Adversarial Attacks

- Q1: ViT-S fine-tuning with LoRA (89.33% test accuracy)
- Q2: FGSM, PGD, BIM adversarial attacks and detectors
- WandB: https://wandb.ai/m25csa010-iit-jodhpur/assignment5-q1
- HuggingFace: https://huggingface.co/JD16112001/vit-lora-cifar100-best

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

# Push to GitHub
git push -u origin "Assignment-5"
```

---

## Step 3: Handle Large Files (if needed)

If you get errors about file size, use Git LFS:

```powershell
# Install Git LFS (run once)
git lfs install

# Track large weight files
git lfs track "*.pt"
git add .gitattributes

# Then commit and push again
git add weights_submit/
git commit -m "Add model weights with LFS"
git push
```

---

## Files to Submit

### Required:
- [x] `requirements.txt`
- [x] All `.py` files in `src/`
- [x] `README.md` with tables, commands, and links
- [x] `reports/M25CSA010_Jyoti_Ass5.tex` (compile to PDF before submission)
- [x] `weights_submit/q1/best_lora_r8_a8_d0.1.pt` (Q1 best model)
- [x] `weights_submit/q2/` (all Q2 weights)
- [x] `results/` (train-val tables and graphs)

### Links to Include:
- **WandB Q1**: https://wandb.ai/m25csa010-iit-jodhpur/assignment5-q1
- **WandB Q2**: https://wandb.ai/m25csa010-iit-jodhpur/assignment5-q2  
- **HuggingFace**: https://huggingface.co/JD16112001/vit-lora-cifar100-best

---

## Verify Before Submitting

Check that your GitHub repo has:
1. Branch named "Assignment-5" (or "Assignment 5")
2. README.md with results tables
3. All Python source files
4. Model weights
5. Results CSV and PNG files
6. LaTeX/PDF report
