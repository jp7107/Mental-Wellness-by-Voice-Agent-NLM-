# MIND EASE - Model Training Guide

## Overview

Your project uses **4 models**, but only **1 requires custom training** for your specific use case. The others are pre-trained and ready to use.

---

## Models Analysis

### 1. ✅ **Whisper (STT)** - NO TRAINING NEEDED
- **File**: `engine/src/stt.cpp`
- **Model**: OpenAI Whisper Small (quantized)
- **Status**: Pre-trained, production-ready
- **Why**: Whisper is already trained on 680k hours of multilingual speech data
- **Action**: Just download the model using `scripts/download_models.sh`

---

### 2. 🔥 **Emotion Classifier** - REQUIRES TRAINING
- **File**: `engine/src/emotion.cpp`
- **Model**: Qwen2.5-1.5B with LoRA adapter (ONNX format)
- **Current State**: Using keyword-based fallback
- **Why Train**: To get accurate emotion detection for mental wellness context
- **Target Emotions**: calm, sad, anxious, angry, fearful, distressed

#### **THIS IS THE MODEL YOU NEED TO TRAIN!**

---

### 3. ✅ **Phi-3 Mini (LLM)** - NO TRAINING NEEDED (Optional Fine-tuning)
- **File**: `engine/src/llm.cpp`
- **Model**: Microsoft Phi-3-mini-4k-instruct (quantized GGUF)
- **Status**: Pre-trained with instruction following
- **Why**: Already trained for empathetic conversation
- **Action**: Download and use as-is
- **Optional**: Fine-tune if you want more specific mental wellness responses

---

### 4. ✅ **Kokoro (TTS)** - NO TRAINING NEEDED
- **File**: Backend TTS service
- **Model**: Kokoro v0.19 (ONNX)
- **Status**: Pre-trained voice synthesis
- **Action**: Download and use

---

## Priority Training Plan

### 🎯 **PRIMARY: Emotion Classifier Training**

This is the **most critical** model to train because:
1. It's currently using a simple keyword fallback
2. Mental wellness requires nuanced emotion detection
3. Your 6-class taxonomy is domain-specific



---

## Step-by-Step: Training the Emotion Classifier

### Architecture
- **Base Model**: Qwen2.5-1.5B (or Qwen2.5-0.5B for faster inference)
- **Method**: LoRA (Low-Rank Adaptation) fine-tuning
- **Output Format**: ONNX for C++ integration
- **Classes**: 6 emotions (calm, sad, anxious, angry, fearful, distressed)

### Step 1: Prepare Training Data

Create a dataset with mental wellness conversations labeled with emotions.

**Dataset Structure** (`training/emotion_dataset.jsonl`):
```json
{"text": "I've been feeling really anxious about work lately", "label": "anxious"}
{"text": "I'm doing okay today, just taking things slow", "label": "calm"}
{"text": "I can't stop crying, everything feels hopeless", "label": "sad"}
{"text": "I'm so frustrated with everything right now", "label": "angry"}
{"text": "I'm terrified something bad is going to happen", "label": "fearful"}
{"text": "I don't want to be here anymore", "label": "distressed"}
```

**Recommended Dataset Sources**:
1. **GoEmotions** (Google) - 58k Reddit comments with emotions
2. **EmotionLines** - Conversational emotion dataset
3. **DailyDialog** - Daily conversations with emotion labels
4. **Custom**: Collect anonymized mental health forum posts (with consent)

**Minimum Data Requirements**:
- Training: 500-1000 examples per class (3000-6000 total)
- Validation: 100-200 examples per class
- Test: 100-200 examples per class

### Step 2: Create Training Script

Create `training/train_emotion_model.py`:

```python
"""
MIND EASE - Emotion Classifier Training
Fine-tunes Qwen2.5 with LoRA for 6-class emotion detection
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Emotion label mapping
LABEL_MAP = {
    "calm": 0,
    "sad": 1,
    "anxious": 2,
    "angry": 3,
    "fearful": 4,
    "distressed": 5
}

ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

def load_and_prepare_data(data_path):
    """Load JSONL dataset and prepare for training"""
    dataset = load_dataset('json', data_files={
        'train': f'{data_path}/train.jsonl',
        'validation': f'{data_path}/val.jsonl',
        'test': f'{data_path}/test.jsonl'
    })
    
    return dataset

def tokenize_function(examples, tokenizer):
    """Tokenize text and convert labels"""
    tokenized = tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128  # Short for real-time inference
    )
    tokenized['labels'] = [LABEL_MAP[label] for label in examples['label']]
    return tokenized

def compute_metrics(eval_pred):
    """Compute accuracy and F1 scores"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # or "Qwen/Qwen2.5-0.5B" for faster
    OUTPUT_DIR = "./emotion_model_output"
    DATA_PATH = "./emotion_dataset"
    
    print(f"[1/6] Loading base model: {MODEL_NAME}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=6,
        id2label=ID2LABEL,
        label2id=LABEL_MAP
    )
    
    # Configure LoRA
    print("[2/6] Configuring LoRA")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,                    # LoRA rank
        lora_alpha=32,           # LoRA scaling
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Qwen2.5 attention layers
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("[3/6] Loading dataset")
    dataset = load_and_prepare_data(DATA_PATH)
    
    # Tokenize
    print("[4/6] Tokenizing")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    print("[5/6] Starting training")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train
    trainer.train()
    
    # Evaluate on test set
    print("[6/6] Evaluating on test set")
    test_results = trainer.evaluate(tokenized_dataset['test'])
    print("\nTest Results:")
    print(test_results)
    
    # Detailed classification report
    predictions = trainer.predict(tokenized_dataset['test'])
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    print("\nClassification Report:")
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=list(LABEL_MAP.keys())
    ))
    
    # Save model
    print(f"\nSaving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n✅ Training complete!")
    print(f"Next step: Export to ONNX using export_to_onnx.py")

if __name__ == "__main__":
    main()
```

### Step 3: Export to ONNX

Create `training/export_to_onnx.py`:

```python
"""
Export trained LoRA model to ONNX format for C++ inference
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_to_onnx(model_path, output_path):
    """Export PyTorch model to ONNX"""
    
    print("[1/4] Loading trained model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Dummy input for tracing
    dummy_text = "I'm feeling anxious"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True
    )
    
    print("[2/4] Exporting to ONNX")
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    print("[3/4] Verifying ONNX model")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print("[4/4] Quantizing for faster inference")
    quantized_path = output_path.replace('.onnx', '_quantized.onnx')
    quantize_dynamic(
        output_path,
        quantized_path,
        weight_type=QuantType.QUInt8
    )
    
    print(f"\n✅ Export complete!")
    print(f"   Full model: {output_path}")
    print(f"   Quantized:  {quantized_path}")
    print(f"\nCopy to: models/qwen2.5-emotion-lora/adapter_model.onnx")

if __name__ == "__main__":
    MODEL_PATH = "./emotion_model_output"
    OUTPUT_PATH = "./emotion_model.onnx"
    export_to_onnx(MODEL_PATH, OUTPUT_PATH)
```

### Step 4: Training Requirements

Create `training/requirements.txt`:

```
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0
datasets>=2.14.0
scikit-learn>=1.3.0
onnx>=1.15.0
onnxruntime>=1.16.0
accelerate>=0.24.0
```

### Step 5: Run Training

```bash
# Create training directory
mkdir -p training/emotion_dataset

# Prepare your dataset (train.jsonl, val.jsonl, test.jsonl)
# ... (collect and format your data)

# Install dependencies
cd training
pip install -r requirements.txt

# Train the model (GPU recommended, ~2-4 hours)
python train_emotion_model.py

# Export to ONNX
python export_to_onnx.py

# Copy to project
cp emotion_model_quantized.onnx ../models/qwen2.5-emotion-lora/adapter_model.onnx
```

### Step 6: Update C++ Integration

The C++ code in `engine/src/emotion.cpp` already has ONNX integration scaffolding. You'll need to complete the inference implementation:

**Key areas to implement** (lines 140-145 in emotion.cpp):
```cpp
#if HAS_ONNX
    if (session_) {
        // TODO: Complete ONNX inference
        // 1. Tokenize text using Qwen2.5 tokenizer
        // 2. Create input tensors (input_ids, attention_mask)
        // 3. Run inference: session_->Run(...)
        // 4. Extract logits and apply softmax
        // 5. Map to EmotionLabel enum
    }
#endif
```



---

## Optional: Fine-tune Phi-3 for Better Responses

If you want more domain-specific mental wellness responses, you can fine-tune Phi-3.

### When to Fine-tune Phi-3:
- ✅ You have 1000+ high-quality mental wellness conversation examples
- ✅ You want responses more aligned with specific therapeutic approaches
- ✅ You need to reduce hallucinations in crisis situations
- ❌ Don't fine-tune if you're satisfied with current template responses

### Quick Fine-tuning Guide:

Create `training/train_llm.py`:

```python
"""
Fine-tune Phi-3 Mini for mental wellness conversations
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def format_conversation(example):
    """Format as Phi-3 chat template"""
    return {
        'text': f"<|system|>\n{example['system']}\n<|end|>\n"
                f"<|user|>\n{example['user']}\n<|end|>\n"
                f"<|assistant|>\n{example['assistant']}\n<|end|>"
    }

def main():
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    
    # Load your conversation dataset
    dataset = load_dataset('json', data_files='wellness_conversations.jsonl')
    dataset = dataset.map(format_conversation)
    
    # Training
    training_args = TrainingArguments(
        output_dir="./phi3_wellness",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        fp16=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train']
    )
    
    trainer.train()
    model.save_pretrained("./phi3_wellness_output")

if __name__ == "__main__":
    main()
```

**Dataset Format** (`wellness_conversations.jsonl`):
```json
{"system": "You are a compassionate mental wellness companion...", "user": "I'm feeling really anxious", "assistant": "I hear you. Anxiety can feel overwhelming. Let's take a moment together. Can you tell me what's weighing on you most right now?"}
```

After training, convert to GGUF format:
```bash
# Use llama.cpp converter
python llama.cpp/convert-hf-to-gguf.py ./phi3_wellness_output --outfile phi3-wellness-q4.gguf --outtype q4_0
```

---

## Training Infrastructure Recommendations

### Hardware Requirements

**Emotion Classifier Training:**
- **Minimum**: 16GB RAM, CPU only (slow, ~8-12 hours)
- **Recommended**: GPU with 8GB+ VRAM (RTX 3060, ~2-4 hours)
- **Optimal**: GPU with 16GB+ VRAM (RTX 4080, A100, ~1-2 hours)

**Phi-3 Fine-tuning (if needed):**
- **Minimum**: GPU with 16GB VRAM
- **Recommended**: GPU with 24GB+ VRAM or multi-GPU setup

### Cloud Options

If you don't have local GPU:

1. **Google Colab Pro** ($10/month)
   - T4 GPU (16GB VRAM)
   - Good for emotion classifier training

2. **Lambda Labs** (~$0.50-1.50/hour)
   - A100 GPUs available
   - Pay-as-you-go

3. **RunPod** (~$0.30-1.00/hour)
   - Various GPU options
   - Good for experimentation

4. **Vast.ai** (~$0.20-0.80/hour)
   - Cheapest option
   - Community GPUs

---

## Evaluation Metrics

### Emotion Classifier Success Criteria:

- **Accuracy**: >75% on test set
- **F1-Score (Macro)**: >0.70
- **Per-class F1**: >0.65 for all emotions
- **Critical**: F1 >0.80 for "distressed" (safety-critical)

### Testing Strategy:

1. **Quantitative**: Run on held-out test set
2. **Qualitative**: Manual review of 100 predictions
3. **Safety**: Test with crisis-related phrases
4. **Edge Cases**: Test with ambiguous emotions

---

## Integration Checklist

After training, complete these steps:

- [ ] Train emotion classifier on mental wellness dataset
- [ ] Export to ONNX format (quantized)
- [ ] Copy to `models/qwen2.5-emotion-lora/adapter_model.onnx`
- [ ] Complete ONNX inference in `engine/src/emotion.cpp`
- [ ] Test with `./mindease_engine --test`
- [ ] Validate accuracy on real conversations
- [ ] Monitor safety escalation triggers
- [ ] (Optional) Fine-tune Phi-3 if needed
- [ ] (Optional) Convert Phi-3 to GGUF and test

---

## Quick Start Commands

```bash
# 1. Create training environment
mkdir -p training/emotion_dataset
cd training
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# 2. Prepare your dataset
# Create train.jsonl, val.jsonl, test.jsonl in emotion_dataset/

# 3. Train
python train_emotion_model.py

# 4. Export
python export_to_onnx.py

# 5. Deploy
cp emotion_model_quantized.onnx ../models/qwen2.5-emotion-lora/adapter_model.onnx

# 6. Test
cd ../engine/build
./mindease_engine --test
```

---

## Summary

### ✅ What You MUST Train:
1. **Emotion Classifier** (Qwen2.5 + LoRA) - This is essential

### ⚠️ What You CAN Train (Optional):
2. **Phi-3 LLM** - Only if you need more specific responses

### ❌ What You DON'T Need to Train:
3. **Whisper STT** - Pre-trained and ready
4. **Kokoro TTS** - Pre-trained and ready

### Files to Create:
- `training/train_emotion_model.py` - Main training script
- `training/export_to_onnx.py` - ONNX export script
- `training/requirements.txt` - Python dependencies
- `training/emotion_dataset/train.jsonl` - Training data
- `training/emotion_dataset/val.jsonl` - Validation data
- `training/emotion_dataset/test.jsonl` - Test data

### Files to Modify:
- `engine/src/emotion.cpp` - Complete ONNX inference (lines 140-145)

---

## Need Help?

Common issues and solutions:

**Q: Where do I get training data?**
A: Use GoEmotions dataset, EmotionLines, or collect from mental health forums (with consent)

**Q: Can I train on CPU?**
A: Yes, but it will be slow (8-12 hours). Use Qwen2.5-0.5B for faster training.

**Q: How much data do I need?**
A: Minimum 500 examples per class (3000 total), ideally 1000+ per class

**Q: What if my accuracy is low?**
A: Try: (1) More training data, (2) Longer training (5 epochs), (3) Larger LoRA rank (r=32)

**Q: Can I use a different base model?**
A: Yes! Try: BERT, RoBERTa, or smaller models like DistilBERT for faster inference

---

**Good luck with training! Focus on the emotion classifier first - it's the heart of your mental wellness system.** 🚀
