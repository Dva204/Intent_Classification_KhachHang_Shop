import joblib 
from preprocess import clean_text

print("Đang tải mô hình ... ")
model = joblib.load("intent_model.joblib")
print("Tải mô hình thành công !!")

LABEL_NAMES = {
    "hoi_gia": "Hỏi giá",
    "tinh_trang_don": "Hỏi tình trạng đơn",
    "doi_tra": "Đổi / trả hàng",
    "size_con_hang": "Hỏi size / còn hàng",
    "khac": "Khác (chào hỏi, cảm ơn,...)"
}

def predict_intent(question):
    #tiền xử lý câu hỏi
    cleaned = clean_text(question)

    #tính xác suất các tác nhân
    probs = model.predict_proba([cleaned])[0]

    #tìm nhãn có xác suất cao nhất
    
    #tìm vị trí lớn nhất
    best_idx = probs.argmax()

    #nhãn dự đoán
    label = model.classes_[best_idx]

    #Độ tn cậy
    confidence = probs[best_idx]

    return cleaned, label, confidence, probs

#Giao diện nhập câu hỏi
if __name__ == "__main__":
    print("\n======================================")
    print("       HỆ THỐNG PHÂN LOẠI Ý ĐỊNH")
    print("======================================")

    print("Nhập exit hoặc quit để thoát.")

    while True:
        question = input(
            "\nNhập câu hỏi: "
        ).strip()

        #kiểm tra lệnh thoát
        if question.lower() in ["exit", "quit"]:
            print("\nĐã thoát chương trình!!")
            break
        
        #bỏ qua nếu user nhập rỗng
        if not question:
            continue

        #Gọi hàm dự đoán
        cleaned, label, confidence, probs = predict_intent(question)

        #In kết quả
        print("\n---------- KẾT QUẢ DỰ ĐOÁN ----------")

        print(f"Câu hỏi gốc: {question}")

        print(f"Câu sau tiền xử lý: {cleaned}")

        print(
            f"Ý định dự đoán: {LABEL_NAMES.get(label, label)}"
        )

        print(
            f"Độ tin cậy: {confidence * 100:.2f}%"
        )

        #cảnh báo nếu độ tin cậy quá thấp
        if confidence < 0.5:
            print(
                "Cảnh báo độ tin cậy thấp, "
                "nên chuyển đến nhân viên tư vấn"
            )

        #Hiển thị xác suất của cả 5 nhãn
        print("\nXác suất các nhãn: ")

        for class_name, probability in zip(model.classes_, probs):
            display_name = LABEL_NAMES.get(
                class_name, class_name
            )
            print(
                f"- {display_name}: {probability * 100:.2f}%"
            )
        print("--------------------------------------") 