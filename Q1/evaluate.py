from sacrebleu import BLEU
import sys

def evaluate_translation():
   
    
    # Read files
    print("Reading files...")
    with open('output.txt', 'r', encoding='utf-8') as f:
        hypothesis = f.read().strip()
    
    with open('reference.txt', 'r', encoding='utf-8') as f:
        reference = f.read().strip()
    
    # Initialize BLEU scorer
    bleu = BLEU()
    
    # Compute BLEU score
    print("Computing BLEU score...")
    score = bleu.corpus_score(hypothesis, [reference])
    
    # Display results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"BLEU Score: {score.score:.4f}")
    print(f"Precisions: {score.precisions}")
    print(f"Brevity Penalty: {score.bp:.4f}")
    print(f"Ratio: {score.ratio:.4f}")
    print("="*60)
    
    # Save results to file
    with open('bleu_score.txt', 'w', encoding='utf-8') as f:
        f.write(f"BLEU Score: {score.score:.4f}\n")
        f.write(f"Details:\n")
        f.write(f"  Precisions: {score.precisions}\n")
        f.write(f"  Brevity Penalty: {score.bp:.4f}\n")
        f.write(f"  Ratio: {score.ratio:.4f}\n")
    
    print("\nResults saved to bleu_score.txt")
    return score.score

if __name__ == "__main__":
    try:
        score = evaluate_translation()
        sys.exit(0)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)
