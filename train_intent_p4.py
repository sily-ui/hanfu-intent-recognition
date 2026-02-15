# train_intent.py - P4 GPU版本 (带可视化评估)
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# --- 1. 设置设备 --- 优先使用GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# --- 2. 加载数据 ---
dataset = load_dataset('json', data_files={'train': 'train.json', 'test': 'test.json'})

# --- 3. 标签映射 ---
label_list = ["design", "compliance_check", "knowledge", "dialogue"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# --- 4. 加载模型和分词器 ---
model_path = "./models/AI-ModelScope/bert-base-chinese"

print("正在从本地加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("正在从本地加载模型...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

model = model.to(device)
print(f"✅ 模型加载成功，设备: {device}, 参数量: {model.num_parameters():,}")

# --- 5. 数据预处理 ---
def preprocess_function(examples):
    tokenized = tokenizer(
        examples['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=128
    )
    tokenized["labels"] = [label2id.get(label, 0) for label in examples["label"]]
    return tokenized

encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['text', 'label'])
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# --- 6. 评估指标函数 ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0)
    }

# --- 7. 训练参数 ---
training_args = TrainingArguments(
    output_dir="./intent-model-p4",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,
    bf16=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    report_to="none",
    logging_dir="./logs",
    remove_unused_columns=False,
    use_cpu=False,
)

# --- 8. 创建Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# --- 9. 开始训练 ---
print("开始训练...")
train_result = trainer.train()

# --- 10. 创建可视化输出目录 ---
os.makedirs("./evaluation_results", exist_ok=True)

# --- 11. 绘制训练Loss曲线 ---
def plot_training_loss(trainer, save_path):
    log_history = trainer.state.log_history
    train_loss = []
    eval_loss = []
    train_epochs = []
    eval_epochs = []
    
    for entry in log_history:
        if 'loss' in entry:
            train_loss.append(entry['loss'])
            train_epochs.append(entry['epoch'])
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
            eval_epochs.append(entry['epoch'])
    
    plt.figure(figsize=(10, 6))
    if train_loss:
        plt.plot(train_epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    if eval_loss:
        plt.plot(eval_epochs, eval_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss curve saved: {save_path}")

plot_training_loss(trainer, "./evaluation_results/loss_curve.png")

# --- 12. 绘制评估指标曲线 ---
def plot_metrics_curve(trainer, save_path):
    log_history = trainer.state.log_history
    metrics_data = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'epochs': []}
    
    for entry in log_history:
        if 'eval_accuracy' in entry:
            metrics_data['accuracy'].append(entry['eval_accuracy'])
            metrics_data['precision'].append(entry.get('eval_precision', 0))
            metrics_data['recall'].append(entry.get('eval_recall', 0))
            metrics_data['f1'].append(entry.get('eval_f1', 0))
            metrics_data['epochs'].append(entry['epoch'])
    
    if metrics_data['epochs']:
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_data['epochs'], metrics_data['accuracy'], 'g-o', label='Accuracy', linewidth=2)
        plt.plot(metrics_data['epochs'], metrics_data['precision'], 'b-s', label='Precision', linewidth=2)
        plt.plot(metrics_data['epochs'], metrics_data['recall'], 'orange', marker='^', label='Recall', linewidth=2)
        plt.plot(metrics_data['epochs'], metrics_data['f1'], 'r-D', label='F1-Score', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Evaluation Metrics Curve', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Metrics curve saved: {save_path}")

plot_metrics_curve(trainer, "./evaluation_results/metrics_curve.png")

# --- 13. 详细评估与混淆矩阵 ---
print("\n" + "="*60)
print("=== 详细评估结果 ===")
print("="*60)

predictions = trainer.predict(encoded_dataset['test'])
preds = predictions.predictions.argmax(axis=-1)
labels = predictions.label_ids

accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average='weighted', zero_division=0)
recall = recall_score(labels, preds, average='weighted', zero_division=0)
f1 = f1_score(labels, preds, average='weighted', zero_division=0)

print(f"\n[Overall Metrics]")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

print(f"\n[Classification Report - Detailed Metrics per Class]")
class_report = classification_report(labels, preds, target_names=label_list, digits=4)
print(class_report)

with open("./evaluation_results/classification_report.txt", 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("Intent Recognition Model Evaluation Report\n")
    f.write("="*60 + "\n\n")
    f.write(f"[Overall Metrics]\n")
    f.write(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall:    {recall:.4f}\n")
    f.write(f"  F1-Score:  {f1:.4f}\n\n")
    f.write(f"[Classification Report]\n")
    f.write(class_report)
    f.write(f"\n[Training Parameters]\n")
    f.write(f"  Learning Rate: {training_args.learning_rate}\n")
    f.write(f"  Batch Size: {training_args.per_device_train_batch_size}\n")
    f.write(f"  Epochs: {training_args.num_train_epochs}\n")
    f.write(f"  Weight Decay: {training_args.weight_decay}\n")
print(f"✅ Classification report saved: ./evaluation_results/classification_report.txt")

# --- 14. 绘制混淆矩阵 ---
def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {save_path}")

plot_confusion_matrix(labels, preds, label_list, "./evaluation_results/confusion_matrix.png")

# --- 15. 绘制每个类别的F1分数柱状图 ---
def plot_f1_per_class(labels, preds, class_names, save_path):
    from sklearn.metrics import f1_score
    f1_scores = f1_score(labels, preds, average=None, zero_division=0)
    
    plt.figure(figsize=(10, 6))
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
    bars = plt.bar(class_names, f1_scores, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Intent Class', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score per Class', fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ F1 score bar chart saved: {save_path}")

plot_f1_per_class(labels, preds, label_list, "./evaluation_results/f1_per_class.png")

# --- 16. 保存评估指标JSON ---
metrics_json = {
    "overall_metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    },
    "per_class_f1": {label: float(f1_score(labels, preds, average=None, zero_division=0)[i]) 
                     for i, label in enumerate(label_list)},
    "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    "training_args": {
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "epochs": training_args.num_train_epochs,
        "weight_decay": training_args.weight_decay
    }
}

with open("./evaluation_results/metrics.json", 'w', encoding='utf-8') as f:
    json.dump(metrics_json, f, ensure_ascii=False, indent=2)
print(f"✅ Metrics JSON saved: ./evaluation_results/metrics.json")

# --- 17. 保存最终模型 ---
final_model_path = "./intent-model-final-p4"
print(f"\nSaving model to: {final_model_path}")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"✅ Model training completed, saved to: {final_model_path}")

# --- 18. 推理示例 ---
print("\n=== Inference Examples ===")
test_texts = [
    "我想设计一件汉服，有什么建议？",
    "唐代的圆领袍和宋代的有什么区别？",
    "今天天气真好，适合穿汉服去玩。",
    "这件衣服上的纹样在古代是不是僭越了？"
]

for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    predicted_class = outputs.logits.argmax().item()
    predicted_label = id2label[predicted_class]
    confidence = probs[predicted_class].item()
    print(f"Input: {text}")
    print(f"Predicted intent: {predicted_label} (confidence: {confidence:.2%})")
    print("---")

print("\n" + "="*60)
print("Training and evaluation completed! All results saved to ./evaluation_results/:")
print("  - loss_curve.png         (Training loss curve)")
print("  - metrics_curve.png      (Evaluation metrics curve)")
print("  - confusion_matrix.png   (Confusion matrix)")
print("  - f1_per_class.png       (F1 score per class)")
print("  - classification_report.txt (Detailed classification report)")
print("  - metrics.json           (Evaluation metrics JSON)")
print("="*60)
