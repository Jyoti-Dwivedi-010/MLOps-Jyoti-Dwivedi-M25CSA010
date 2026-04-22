from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

def translate_text(text, model_name="Helsinki-NLP/opus-mt-bn-en"):
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate translation
    translated = model.generate(**inputs)
    
    # Decode output
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return translated_text

def main():
    
    # Read input file
    print("Reading input file...")
    with open('input.txt', 'r', encoding='utf-8') as f:
        input_text = f.read().strip()
    
    # Split into sentences
    sentences = input_text.split('\n')
    print(f"Found {len(sentences)} sentences to translate")
    
    # Translate sentences
    print("Starting translation...")
    translated_sentences = []
    
    for i, sentence in enumerate(sentences, 1):
        if sentence.strip():
            print(f"Translating sentence {i}/{len(sentences)}...")
            try:
                translated = translate_text(sentence)
                translated_sentences.append(translated)
            except Exception as e:
                print(f"Error translating sentence {i}: {e}")
                translated_sentences.append(sentence)  # Keep original on error
        else:
            translated_sentences.append(sentence)
    
    # Write output
    output_text = '\n'.join(translated_sentences)
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(output_text)
    
    print(f"\n✓ Translation complete!")
    print(f"Output saved to output.txt")
    
    # Print first translated sentence
    print("\n" + "="*50)
    print("FIRST TRANSLATED STATEMENT:")
    print("="*50)
    if translated_sentences:
        print(translated_sentences[0])
    print("="*50)

if __name__ == "__main__":
    main()
