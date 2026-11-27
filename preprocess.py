import re

ABBREVIATIONS = {
        "ko": "không",
        "mik": "mình",
        "dc": "được",
        "bn": "bao nhiêu",
        "nhiu": "nhiêu",
        "sp": "sản phẩm",
        "z": "vậy",
        "hong": "không",
        "k": "không",
        "đc": "được",
        "mk": "mình",
        "sz": "size",
        "b": "bạn",
        "r": "rồi"
    }

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\b[a-z]{2,3}\d{4,6}\b', 'ma_don', text)
    text = re.sub(r'[^\w\s]', ' ', text)

    words = text.split()

    for i,word in enumerate(words):
        if word == "m":
            if i == 0 or words[i - 1] != "size":
                words[i] = "mình"
        else:
            words[i] = ABBREVIATIONS.get(word, word)
    
    text = " ".join(words)

    return text

if __name__ == "__main__":
    test_sentences = [
        "Cho minh hoi tinh trang don SP32365",
        "túi xách màu trắng còn hàng ko bạn?",
        "MD921822 giao den dau roi b?",
        "mik muốn hỏi giá sản phẩm này bao nhiu",
        "dép này còn size M k ạ?",
        "Ừm để mik hỏi lại ba mẹ r báo lại shop"
    ]

    for sentence in test_sentences:
        print("Gốc: ", sentence)
        print("Sau xử lý:", clean_text(sentence))
        print("-" * 50)