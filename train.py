import torch
import config
from dataset import Dataset
from model import ResNet9
from engine import train_fn, eval_fn

cifar10 = Dataset()
# cifar10.download_dataset()
train_dataloader, valid_dataloader = cifar10.get_dataloader()
model = ResNet9(3, 10)

if torch.cuda.is_available():
    device = torch.device('gpu')
else:
    device = torch.device('cpu')

model = model.to(device)
steps = len(train_dataloader) * config.EPOCHS
optimizer = torch.optim.Adam(model.parameters(), config.LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, config.LEARNING_RATE, epochs=config.EPOCHS, total_steps=steps)

for epoch in range(config.EPOCHS):
    avg_loss = train_fn(train_dataloader, model, device, optimizer, scheduler)
    avg_acc = eval_fn(valid_dataloader, model, device)
    print(f'Epoch: {epoch} Avg train loss: {avg_loss} Avg valid acc: {avg_acc}')
