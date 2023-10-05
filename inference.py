import torch
from models import  PhoBertClassifierV1
from transformers import AutoTokenizer

# Initialize model and load trained weights
model = PhoBertClassifierV1()
model.load_state_dict(torch.load('/kaggle/working/PhoBertTuneV4.pth'))
model.eval()

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Feature names for each classifier
feature_names = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']

# Test on custom text
custom_text = "Được đặt trên một chiếc tàu nổi giữa đầm mang tới không gian thơ mộng, tận hưởng trọn vẹn những làn gió biển mát lành và nhân viên mặc đồng phục quần tây - áo trắng vô cùng lịch sự, nhà hàng Hoa Hoa (còn có tên gọi khác là Tàu Hoa Hoa) là một trong những quán hải sản ngon ở Quy Nhơn mà bạn nhất định phải thử. Dĩ nhiên không chỉ được biết đến vì thiết kế độc đáo mà đồ ăn tại đây cũng vô cùng ngon và đa dạng, nhất là các món về cá mú và sứa."  # Replace with your text
encoded = tokenizer(custom_text, return_tensors='pt', padding='max_length', max_length=256, truncation=True)
input_ids = encoded['input_ids'].to(device)
attention_mask = encoded['attention_mask'].to(device)

# Get predictions
with torch.no_grad():
    predictions = model(input_ids, attention_mask)

# Convert predictions to probabilities (assuming Softmax was used)
probabilities = [pred.softmax(dim=1) for pred in predictions]

# Print the probabilities and the class with the highest probability
for i, probs in enumerate(probabilities):
    max_prob, max_idx = torch.max(probs, dim=1)
    print(f"Feature: {feature_names[i]}, Probabilities: {probs}, Highest Probability Class: {max_idx.item()} with probability {max_prob.item()}")
