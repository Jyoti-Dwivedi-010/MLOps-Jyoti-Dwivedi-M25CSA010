import torch
import optuna
import torch.quantization
from datasets import load_dataset
from speechbrain.inference.speaker import EncoderClassifier

# Import our custom modules
from metrics import calculate_gflops, evaluate_accuracy
from qat_trainer import run_qat_epoch

def main():
    print("Loading Model and Dataset...")
    
    # Load base model from SpeechBrain
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="tmp_model"
    )
    base_model = classifier.mods.embedding_model
    base_model.eval()

    # Load Dataset (Using a smaller subset for assignment speed)
    dataset = load_dataset("s3prl/superb", "si")
    val_data = dataset['validation']
    test_data = dataset['test']

    # Dummy input for GFLOPs: Batch=1, Time=300 (frames), Features=80 (Fbank bins)
    dummy_input = torch.randn(1, 300, 80)

    # ==========================================
    # TASK 1: Baseline
    # ==========================================
    print("\n--- TASK 1: Baseline ---")
    baseline_gflops = calculate_gflops(base_model, dummy_input)
    # Using 200 samples for time efficiency. Remove limit=200 for full dataset evaluation.
    baseline_acc = evaluate_accuracy(classifier, base_model, test_data, num_samples=200)

    print(f"Baseline Accuracy: {baseline_acc * 100:.2f}%")
    print(f"Baseline GFLOPs: {baseline_gflops:.4f}")

    # ==========================================
    # TASK 2 & 3: Post-Training Quantization (PTQ)
    # ==========================================
    print("\n--- TASK 2 & 3: Post-Training Quantization ---")
    ptq_model = torch.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8
    )

    ptq_gflops = calculate_gflops(ptq_model, dummy_input)
    gflops_saved = baseline_gflops - ptq_gflops
    ptq_acc = evaluate_accuracy(classifier, ptq_model, test_data, num_samples=200)
    acc_drop = baseline_acc - ptq_acc

    print(f"PTQ Accuracy: {ptq_acc * 100:.2f}% (Drop of {acc_drop * 100:.2f}%)")
    print(f"PTQ GFLOPs: {ptq_gflops:.4f} (Saved {gflops_saved:.4f} GFLOPs)")

    # ==========================================
    # TASK 4: QAT with Optuna
    # ==========================================
    print("\n--- TASK 4: QAT via Optuna (Running 4 Trials) ---")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        
        # Reset model for QAT
        qat_model = classifier.mods.embedding_model
        qat_model.train()
        
        # Set quantization backend (use 'qnnpack' if on ARM/Mac, 'fbgemm' for x86 Linux Server)
        qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(qat_model, inplace=True)
        
        optimizer = torch.optim.Adam(qat_model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Run a short training loop
        qat_model = run_qat_epoch(qat_model, classifier, val_data, optimizer, limit=50)
        
        # Convert to quantized model for evaluation
        qat_model.eval()
        quantized_final = torch.quantization.convert(qat_model.cpu(), inplace=False)
        
        acc = evaluate_accuracy(classifier, quantized_final, test_data, num_samples=100)
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=4)

    print(f"Best Hyperparameters: {study.best_params}")
    best_qat_acc = study.best_value

    # ==========================================
    # TASK 5: Final Analysis
    # ==========================================
    print("\n--- TASK 5: Final Analysis ---")
    final_acc_diff = baseline_acc - best_qat_acc

    print(f"Best QAT Model Accuracy: {best_qat_acc * 100:.2f}%")
    print(f"Final Total Performance Difference: {final_acc_diff * 100:.2f}% absolute accuracy difference")
    print(f"Final Total GFLOPs Saved: {gflops_saved:.4f} GFLOPs")

if __name__ == "__main__":
    main()