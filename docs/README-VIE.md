# Chord Practice System

[English](../README.md) · **Tiếng Việt**

Hệ thống phân tích bản thu để tạo hai dòng thời gian độc lập cho người luyện nhạc: hợp âm và ranh giới giữa các phần của bài. Đã hoàn tất bước kiểm dữ liệu (P0). Huấn luyện và đánh giá mô hình là các bước tiếp theo. [Model Architecture Plan v1.2, mục 7 và 13](https://drive.google.com/file/d/1gmz-JaN6I_5oCi65DYxmTpX5nlbG2_tO/view)

## Khởi động trên Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Chọn interpreter trong `.venv` ở IDE. Đặt annotation chi tiết và Chordino features của McGill Billboard trong `data/raw/`. Khi chuyển lên Kaggle, truyền đường dẫn dữ liệu và đầu ra qua cấu hình để dùng lại cùng mã nguồn.

## Chạy P0 trên một bản thu

Từ thư mục gốc, chạy `python src/acr/audit.py 0003 --output runs/p0-0003` hoặc mở [notebook kiểm tra](../notebooks/00_data_audit.ipynb) trong Jupyter. Lệnh Python chỉ dùng thư viện chuẩn. Demo đọc một bản thu trực tiếp trong các gói chính thức. Xem [hướng dẫn P0](p0/P0_INSTRUCTION.md) để tải dữ liệu và chạy toàn bộ corpus.

Mở `timeline.svg` trong thư mục đầu ra để xem phạm vi các khoảng nhãn và frame. `frames.csv` giữ đủ 24 giá trị chroma và gán nhãn theo timestamp, tính cả đầu khoảng và loại cuối khoảng. Nhãn trống là frame ngoài vùng chú giải. `N` là nhãn đã chú giải cho đoạn không có hợp âm. Số liệu tổng hợp nằm trong `summary.json`.

Đã hoàn tất P0 trên CPU cho toàn bộ corpus. Xem [báo cáo P0](p0/P0_REPORT.md) để kiểm tra nguồn dữ liệu, tiêu chí loại, bộ nhãn và lệnh chạy lại. P1 sẽ dùng [danh sách chia tập đã cố định](../splits/billboard-v1.csv).

## Huấn luyện P1

Đã kiểm chứng baseline đầu tiên trên Kaggle, đạt 57,11% frame hợp âm đúng trên validation. Xem [báo cáo P1](p1/P1_REPORT.md) để đọc kết quả và hạn chế với hợp âm hiếm. Xem [hướng dẫn P1](p1/P1_INSTRUCTION.md) và [notebook Kaggle](../notebooks/01_teacher_f.ipynb). Dùng tên thư mục mới trong `runs/` cho mỗi lần chạy.

## P2 · Event teacher với biên chuẩn

Đã có teacher E và mốc so sánh F cố định. Xem [hướng dẫn P2](p2/P2_INSTRUCTION.md), [báo cáo P2](p2/P2_REPORT.md) và [notebook](../notebooks/02_teacher_e_oracle.ipynb). Kết quả full Kaggle đang chờ chạy.

## Bố cục

| Đường dẫn | Nội dung |
|---|---|
| `docs/` | Hướng dẫn và báo cáo bằng Markdown. |
| `reports/p0/` | Các bảng và số liệu P0 đã chốt, lưu trên Git. |
| `src/acr/` | Reader, căn thời gian, ánh xạ nhãn, mô hình và đánh giá. Thêm từng module khi đến mốc tương ứng. |
| `notebooks/` | Các notebook dự kiến: 00_data_audit, 01_teacher_f, 02_teacher_e_oracle. Logic dùng chung nằm trong src. |
| `configs/`, `splits/` | Cấu hình từng thí nghiệm và manifest chia tập theo nhóm bài hát. |
| `data/raw/`, `data/processed/` | Dữ liệu gốc và cache cục bộ. |
| `runs/<run_id>/` | Cấu hình, Git commit, phiên bản dữ liệu, metrics, hình và checkpoint của một lần chạy. |

## Các mốc triển khai

1. **P0 · Kiểm dữ liệu.** Ghép annotation với `bothchroma.csv` theo recording ID và timestamp trên một bản thu. Vẽ timeline để kiểm tra căn chỉnh trước khi chạy toàn bộ corpus. Gom các bản trùng theo bài, cố định split, rồi đếm số event, thời lượng và số bài độc lập của từng quality trên train trước augmentation. Dùng kết quả để chốt vocabulary và tiêu chí hiếm.
2. **P1 · Frame teacher F.** Huấn luyện TCN nhỏ để overfit vài bài train nhằm kiểm tra pipeline. Sau đó train trên phần có nhãn và đánh giá trên bài validation chưa thấy. Lưu điểm theo quality, support, timeline, thời gian chạy và bộ nhớ.
3. **P2 · Event teacher E với biên chuẩn.** Huấn luyện E với trọng số riêng, lấy trung bình hidden features trong từng khoảng chuẩn. Khi đánh giá, gộp xác suất của F trên cùng khoảng với E. Đếm bốn trường hợp cả hai đúng, chỉ F đúng, chỉ E đúng, cả hai sai theo quality và thành phần hợp âm.
4. **P3 · Biên dự đoán.** Huấn luyện boundary head của E. Lặp lại so sánh theo quy tắc ghép event đã cố định và đo mức suy giảm so với P2.
5. **Student.** So sánh S0 (chỉ nhãn người), S1 (một teacher), S2 (hai teacher đồng thuận toàn hợp âm), S3 (đồng thuận từng thành phần). Giữ cố định ngân sách nhãn, tập dữ liệu và năng lực student. Chỉ số chính là macro event recall trên các quality hiếm, chấm đúng toàn hợp âm. S4 selective KD và thay encoder là thí nghiệm tùy chọn.

Nhánh MSA tạo các đoạn luyện tập độc lập với biên đổi hợp âm. Demo ghi rõ mỗi timeline lấy từ annotation hay dự đoán.

## Commit theo bước

Mỗi commit có một mục đích có thể kiểm tra. Ví dụ: `[DOCS] document the Billboard split protocol`, `[FEAT] align Billboard features for one recording`, `[FIX] handle the final chord interval`. Ghi lại mã commit cho mỗi lần chạy thí nghiệm.

## Ngôn ngữ tài liệu

Tiếng Anh là ngôn ngữ chính của dự án. Đặt tài liệu mới trong `docs/`, nhóm theo chủ đề khi cần. Bản dịch tiếng Việt dùng hậu tố `-VIE.md` và liên kết về bản tiếng Anh. Cập nhật hai bản README trong cùng commit khi hướng dẫn hoặc kế hoạch thay đổi.
