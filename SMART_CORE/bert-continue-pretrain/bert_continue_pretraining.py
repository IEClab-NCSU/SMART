from transformers import BertModel, BertConfig, BertTokenizer, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, LineByLineTextDataset, Trainer, \
    TrainingArguments

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

max_length = 512

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='raw-text-for-bert.txt',
    block_size=512
    )

print('No. of lines: ', len(dataset))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir='/continue_pretrain',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
    # prediction_loss_only=True
)

trainer.train()
trainer.save_model('continue_pretrain/')
