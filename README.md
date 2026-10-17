# NLP_LAB - MLP, Transformer, LLama từ đầu

Repository này dùng để tự xây dựng một MLP (Multi-Layer Perceptron) cho MNIST theo hướng học từ nền tảng, thay vì chỉ gọi sẵn thư viện high-level. Mục tiêu chính không phải là làm cho code ngắn nhất, mà là hiểu rõ luồng dữ liệu, công thức toán, và cách các tầng trong mạng học được từ dữ liệu.

## Mục tiêu của project

- Đọc dữ liệu MNIST từ file IDX thô.
- Chuẩn hóa ảnh về vector 784 chiều.
- Xây dựng MLP nhiều tầng bằng NumPy.
- Tự cài forward propagation, loss, backpropagation và update trọng số.
- Theo dõi loss, accuracy và vẽ kết quả huấn luyện.

## Hướng đi khi code MLP

Khi tự code MLP, nên đi theo đúng thứ tự sau:

1. **Xử lý dữ liệu trước**
   - Đọc đúng format MNIST.
   - Tách rõ image và label.
   - Chuẩn hóa dữ liệu về cùng thang đo.
   - Đổi ảnh từ dạng `28 x 28` sang vector `784`.

2. **Xây phần tính toán cơ bản**
   - Viết ReLU.
   - Viết softmax ổn định số học.
   - Viết cross-entropy.
   - Viết accuracy.

3. **Thiết kế kiến trúc mạng**
   - Chọn số input, hidden layer, output.
   - Khởi tạo weight bằng He initialization.
   - Khởi tạo bias bằng 0.
   - Giữ cấu hình trong một class riêng để dễ chỉnh.

4. **Viết forward pass**
   - Mỗi tầng làm `z = a @ W + b`.
   - Hidden layer dùng ReLU.
   - Output layer dùng softmax.
   - Lưu lại `activations` và `pre_activations` để dùng cho backprop.

5. **Viết backward pass**
   - Bắt đầu từ đạo hàm của softmax + cross entropy.
   - Truyền gradient ngược qua từng tầng.
   - Dùng đạo hàm ReLU cho hidden layer.
   - Tính gradient cho cả `W` và `b`.

6. **Cập nhật tham số**
   - Dùng gradient descent hoặc mini-batch gradient descent.
   - Mỗi batch: forward -> loss -> backward -> update.

7. **Huấn luyện và quan sát**
   - Chia dữ liệu thành batch.
   - Shuffle mỗi epoch.
   - Theo dõi loss và accuracy theo epoch.
   - Kiểm tra tập test sau khi train xong.

## Cách tư duy khi tự viết MLP

### 1. Nghĩ theo luồng dữ liệu

Đừng bắt đầu từ code ngay. Hãy trả lời 3 câu hỏi trước:

- Input đi vào có shape gì?
- Qua mỗi tầng shape đổi như thế nào?
- Output cuối cùng cần gì để tính loss và dự đoán?

Nếu không rõ shape thì backprop rất dễ sai.

### 2. Tách bài toán thành các khối nhỏ

MLP không nên viết như một khối lớn. Hãy tách thành:

- đọc dữ liệu
- hàm kích hoạt
- hàm loss
- forward pass
- backward pass
- update
- train loop
- plot kết quả

Mỗi khối nên chạy đúng độc lập trước khi ghép lại.

### 3. Luôn kiểm tra shape

Đây là nguyên tắc quan trọng nhất khi code MLP bằng NumPy.

Ví dụ cần tự hỏi:

- `x_batch` là `(batch_size, 784)` hay chưa?
- `W` của layer 1 là `(784, hidden_dim)` hay chưa?
- `probs` là `(batch_size, 10)` hay chưa?
- `y_one_hot` có cùng batch size với `probs` hay không?

Chỉ cần lệch một shape là phép nhân ma trận sẽ hỏng ngay.

### 4. Ưu tiên công thức trước code

Trước khi code gradient, nên viết công thức ra giấy:

- `z = a @ W + b`
- `a = ReLU(z)`
- `p = softmax(z)`
- `L = cross_entropy(p, y)`

Sau đó mới chuyển từng công thức sang NumPy.

### 5. Đừng tối ưu sớm

Với bài MLP từ đầu, ưu tiên đầu tiên là đúng, không phải nhanh.

- Code rõ ràng hơn code ngắn.
- Có thể lưu thêm biến trung gian để debug.
- In shape ra nếu thấy kết quả lạ.
- Chỉ tối ưu sau khi mô hình chạy đúng.

## Đọc file `MLP_myself.py` theo từng phần

### Phần dữ liệu

Các hàm `read_idx_imgs()` và `read_idx_labels()` xử lý định dạng IDX của MNIST.

Tư duy ở phần này là:

- đọc đúng header
- đọc đúng kiểu dữ liệu
- trả về đúng shape

Nếu phần dữ liệu sai thì toàn bộ mô hình phía sau đều sai theo.

### Phần hàm toán học

Các hàm như `relu_activation()`, `softmax()`, `cross_entropy()` là nền tảng để model học được.

Tư duy ở đây là:

- ReLU dùng cho hidden layer để tạo phi tuyến.
- Softmax dùng cho output nhiều lớp.
- Cross-entropy đo độ lệch giữa dự đoán và nhãn thật.

### Phần lớp `MLP`

Lớp `MLP` nên được xem như một pipeline gồm 4 bước:

- khởi tạo tham số
- forward
- backward
- update

Nếu muốn mở rộng sau này, bạn chỉ cần thay ở 4 bước này.

### Phần train loop

Trong `fit()`:

- shuffle dữ liệu mỗi epoch
- chia batch
- forward
- tính loss
- backward
- cập nhật tham số
- lưu lịch sử học

Đây là vòng lặp trung tâm của toàn bộ mô hình.

## Cách debug khi code MLP

Nếu model lỗi, hãy debug theo thứ tự này:

1. Kiểm tra file dữ liệu và đường dẫn.
2. Kiểm tra shape sau khi load dữ liệu.
3. Kiểm tra forward pass có ra shape đúng không.
4. Kiểm tra loss có ra số hữu hạn không.
5. Kiểm tra backward pass có cùng shape với weight/bias không.
6. Kiểm tra update có làm tham số thay đổi không.
7. Kiểm tra accuracy và plot sau cùng.

## Những lỗi hay gặp

- Đọc nhầm file ảnh và file nhãn.
- Sai header IDX.
- Dùng `set` thay vì `tuple` cho hidden layers.
- Softmax không ổn định số học.
- One-hot sai shape.
- Nhầm giữa `dZ`, `dA`, `dW`, `dB`.
- Vẽ biểu đồ bằng một số nguyên thay vì một dãy giá trị.
- Bỏ quên bước flatten ảnh trước khi đưa vào dense layer.

## Mục tiêu học được sau bài này

Sau khi làm xong MLP này, bạn nên hiểu được:

- dữ liệu vào đi như thế nào qua từng tầng
- vì sao cần activation function
- loss function hoạt động ra sao
- gradient được truyền ngược thế nào
- mini-batch training hoạt động thế nào
- cách debug một model NumPy từ đầu

## Gợi ý mở rộng

Nếu muốn nâng cấp project, bạn có thể làm tiếp:

- thêm nhiều hidden layer hơn
- thêm momentum hoặc Adam
- thêm early stopping
- tách code thành các file module riêng
- viết test cho từng hàm toán học
- log kết quả train ra file

## Cách chạy

```bash
python MLP/MLP_myself.py
```

Nếu môi trường đúng và dữ liệu MNIST nằm trong thư mục `MLP/`, script sẽ đọc dữ liệu, train model và vẽ loss/accuracy.
