import pandas as pd
from model import NaiveBayes
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, test_data):
    """Đánh giá mô hình trên tập test"""
    y_true = []
    y_pred = []
    
    for index, row in test_data.iterrows():
        message = row['SMS']
        expected = row['Label'].lower()
        result = model.classify(message)
        predicted = "ham" if result else "spam"
        
        y_true.append(expected)
        y_pred.append(predicted)
    
    # Tính toán các chỉ số đánh giá
    precision = precision_score(y_true, y_pred, pos_label='spam')
    recall = recall_score(y_true, y_pred, pos_label='spam')
    f1 = f1_score(y_true, y_pred, pos_label='spam')
    conf_matrix = confusion_matrix(y_true, y_pred, labels=['spam', 'ham'])
    
    # In kết quả
    print("\nKết quả đánh giá mô hình:")
    print("-" * 50)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Spam    Ham")
    print(f"Actual Spam    {conf_matrix[0][0]:<8d}{conf_matrix[0][1]:<8d}")
    print(f"       Ham     {conf_matrix[1][0]:<8d}{conf_matrix[1][1]:<8d}")

def main():
    # Tải mô hình đã được huấn luyện
    try:
        model = NaiveBayes.load_model('spam_classifier.pkl')
    except FileNotFoundError:
        print("Không tìm thấy file mô hình. Vui lòng chạy train.py trước!")
        return

    # Đọc và đánh giá trên tập test
    try:
        # Đọc dữ liệu test với xử lý lỗi
        test_data = pd.read_csv('test_sms_vi.csv', encoding='utf-8', on_bad_lines='skip')
        
        # Làm sạch dữ liệu: xóa các dòng có giá trị thiếu và đảm bảo tên cột đúng
        test_data = test_data.dropna()
        test_data.columns = ['Label', 'SMS']
        
        # Chuyển đổi nhãn thành chữ thường để đồng nhất
        test_data['Label'] = test_data['Label'].str.lower()
        
        evaluate_model(model, test_data)
    except FileNotFoundError:
        print("Không tìm thấy file test_sms_vi.csv")
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")
    
    # Demo với tin nhắn mới
    print("\nNhập tin nhắn để test (nhập 'q' để thoát):")
    print("-" * 50)
    while True:
        message = input("\nTin nhắn: ")
        if message.lower() == 'q':
            break
            
        result = model.classify(message)
        print(f"Kết quả: {'HAM' if result else 'SPAM'}")

if __name__ == "__main__":
    main() 