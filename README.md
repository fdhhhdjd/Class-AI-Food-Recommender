# AI-Recommend — Hệ thống gợi ý món ăn (FastAPI + Hugging Face)

Demo hệ thống AI gợi ý món ăn/đồ uống giống KFC / Lotteria / Highlands.  
Dùng mô hình **sentence-transformers/all-MiniLM-L6-v2** để hiểu ngữ nghĩa mô tả món,  
FastAPI làm backend, frontend là HTML/CSS/JS đơn giản.

![Demo](/assets/AI.png)


---

## 🚀 Chức năng chính
- Hiểu mô tả món bằng embedding (AI hiểu "gà rán" giống "Coca").
- Gợi ý món tương tự dựa trên các món người dùng chọn.
- Boost theo quan hệ “món đi kèm" (`pair`).
- Boost theo category (food/drink/dessert).
- Cache embedding để chạy nhanh như real system.
- Giao diện chọn món đẹp, trực quan.

---

## 📁 Cấu trúc project

```
AI-Recomment/
├─ app/
│  ├─ controllers/
│  ├─ models/
│  ├─ routes/
│  ├─ services/
│  ├─ utils/
│  └─ config.py
├─ data/
│  ├─ items.json
│  └─ items_with_vecs.json (auto generate)
├─ web/
│  └─ index.html
├─ server.py
├─ Makefile
├─ README.md
└─ requirements.txt

```

---

## ⚙️ Cài đặt backend

### 1. Copy `.env.example` → `.env`
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
HF_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_PROVIDER=hf-inference
PORT=8000
```

---

### 2. Cài dependencies

```
cd backend
make install
```

Hoặc thủ công:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Chạy server

```
make run-server
```

API Docs:

```
http://127.0.0.1:8000/docs
```

---

## 🧠 Precompute embedding (khuyến nghị)

Tạo file cache để tăng tốc:

```
curl -X POST http://127.0.0.1:8000/api/precompute
```

---

## 📡 Danh sách API

### 1) GET `/api/items`

Trả danh sách món.

```
curl http://127.0.0.1:8000/api/items
```

---

### 2) POST `/api/recommend`

Body:

```json
{
  "history": [1, 4],
  "top": 3,
  "use_cache": true,
  "category_boost": 1.0,
  "pair_boost": 0.15
}
```

Ví dụ:

```
curl -X POST http://127.0.0.1:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"history":[1], "top":3, "use_cache":true}'
```

---

## 🧠 Mô hình AI hoạt động thế nào?

1️⃣ Mô tả món ăn được đưa vào model  
**`sentence-transformers/all-MiniLM-L6-v2`**  
→ Trả về vector embedding 384 chiều.

2️⃣ Lấy trung bình embedding của các món người dùng chọn → tạo **profile khẩu vị**.

3️⃣ So sánh profile với toàn bộ món bằng **cosine similarity**.

4️⃣ Áp dụng **boost**:  
- Món có trong `pair` → cộng điểm  
- Món cùng `category` → cộng điểm  
- Tăng độ chính xác giống hệ thống gợi ý thật.

5️⃣ Trả về danh sách gợi ý.

---

## 🎨 Frontend Demo

Mở:

```
web/index.html
```

Tính năng:
- Chọn món → highlight
- Nút xoá giỏ món đã chọn
- Hiển thị ảnh, giá, mô tả
- Gọi API `/recommend`

---
## 📚 Dạy Học Online

Bên cạnh tài liệu miễn phí, mình còn mở các khóa học online:

- **Lập trình web cơ bản → nâng cao**
- **Ứng dụng về AI và Automation**
- **Kỹ năng phỏng vấn & xây CV IT**

### Thông Tin Đăng Ký

- 🌐 Website: [https://profile-forme.com](https://profile-forme.com)
- 📧 Email: nguyentientai10@gmail.com
- 📞 Zalo/Hotline: 0798805741

---

## 💖 Donate Ủng Hộ

Nếu bạn thấy các source hữu ích và muốn mình tiếp tục phát triển nội dung miễn phí, hãy ủng hộ mình bằng cách donate.  
Mình sẽ sử dụng kinh phí cho:

- 🌐 Server, domain, hosting
- 🛠️ Công cụ bản quyền (IDE, plugin…)
- 🎓 Học bổng, quà tặng cho cộng đồng

### QR Code Ngân Hàng

Quét QR để ủng hộ nhanh:

<img src="https://res.cloudinary.com/ecommerce2021/image/upload/v1760680573/abbank_yjbpat.jpg" alt="QR Code ABBank" width="300">


**QR Code ABBank**  
- Chủ tài khoản: Nguyễn Tiến Tài  
- Ngân hàng: NGAN HANG TMCP AN BINH  
- Số tài khoản: 1651002972052

---

## 📞 Liên Hệ
- 📚 Tiktok Dạy Học: [@code.web.khng.kh](https://www.tiktok.com/@code.web.khng.kh)
- 💻 GitHub: [fdhhhdjd](https://github.com/fdhhhdjd)
- 📧 Email: [nguyentientai10@gmail.com](mailto:nguyentientai10@gmail.com)

Cảm ơn bạn đã quan tâm & chúc bạn học tập hiệu quả! Have a nice day <3!!
