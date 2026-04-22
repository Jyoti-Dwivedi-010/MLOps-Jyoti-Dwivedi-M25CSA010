import torch
from thop import profile
from speechbrain.inference.speaker import EncoderClassifier

def calculate_gflops(target_model, input_tensor):
    """
    Calculates the GFLOPs of a model given a dummy input tensor.
    """
    # Profile the model. verbose=False suppresses thop's print statements
    macs, params = profile(target_model, inputs=(input_tensor, ), verbose=False)
    
    # 1 Multiply-Accumulate (MAC) operation is roughly 2 FLOPs
    gflops = (macs * 2) / 1e9 
    return gflops

def evaluate_accuracy(classifier, embedding_model, dataset, num_samples=None):
    """
    Evaluates Top-1 Identification Accuracy on the provided dataset.
    """
    embedding_model.eval()
    correct = 0
    total = 0
    
    # Limit samples for faster testing during Optuna trials
    eval_data = dataset.select(range(num_samples)) if num_samples else dataset
    
    with torch.no_grad():
        for row in eval_data:
            audio_array = row['audio']['array']
            true_label = row['label']
            
            # Convert raw audio to tensor (Batch x Time)
            wav_tensor = torch.tensor(audio_array).unsqueeze(0).float()
            
            # 1. Use the classifier's feature extractor (compute Fbanks)
            features = classifier.mods.compute_features(wav_tensor)
            features = classifier.mods.mean_var_norm(features, torch.ones(1))
            
            # 2. Pass features through the optimized/quantized embedding model
            embeddings = embedding_model(features)
            
            # 3. Pass embeddings through the classifier head
            predictions = classifier.mods.classifier(embeddings).squeeze()
            
            # Get the predicted class ID
            predicted_label = torch.argmax(predictions).item()
            
            if predicted_label == true_label:
                correct += 1
            total += 1
            
    return correct / total