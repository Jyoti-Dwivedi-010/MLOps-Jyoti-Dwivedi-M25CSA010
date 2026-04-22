import torch
import torch.quantization

def run_qat_epoch(model, classifier, train_data, optimizer, limit=100):
    """
    A short training loop to simulate Quantization-Aware Finetuning.
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    # Select a subset of data to make trials fast
    subset = train_data.select(range(limit))
    
    for row in subset:
        audio_array = row['audio']['array']
        true_label = torch.tensor([row['label']]).long()
        wav_tensor = torch.tensor(audio_array).unsqueeze(0).float()
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            features = classifier.mods.compute_features(wav_tensor)
            features = classifier.mods.mean_var_norm(features, torch.ones(1))
            
        # Forward pass through QAT model
        embeddings = model(features)
        predictions = classifier.mods.classifier(embeddings).squeeze(1)
        
        loss = criterion(predictions, true_label)
        loss.backward()
        optimizer.step()
        
    return model