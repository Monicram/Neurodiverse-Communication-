# train_gec.py
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", required=True)
parser.add_argument("--output_dir", default="t5-gec")
parser.add_argument("--epochs", type=int, default=3)
args = parser.parse_args()

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# load a CSV with 'input_text' and 'target_text'
dataset = load_dataset("csv", data_files={"train": args.train_csv})
def preprocess(examples):
    inputs = ["correct: " + s for s in examples['input_text']]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)
    labels = tokenizer(examples['target_text'], max_length=256, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset["train"].map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=8,
    num_train_epochs=args.epochs,
    logging_steps=50,
    save_total_limit=2,
    fp16=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(args.output_dir)
