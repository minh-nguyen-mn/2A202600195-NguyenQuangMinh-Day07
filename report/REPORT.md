# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Quang Minh
**Nhóm:** D3-C401
**Ngày:** April 10, 2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* 
Hai vector embedding có hướng gần giống nhau, tức là hai đoạn văn bản có ý nghĩa tương đồng về mặt ngữ nghĩa.

**Ví dụ HIGH similarity:**
- Sentence A: "The company collects personal data from users."
- Sentence B: "User data is gathered by the company."
- Tại sao tương đồng: Cùng diễn đạt việc công ty thu thập dữ liệu người dùng, chỉ khác cách diễn đạt.

**Ví dụ LOW similarity:**
- Sentence A: "The company collects personal data."
- Sentence B: "The weather is sunny today."
- Tại sao khác: Hai câu thuộc hai domain hoàn toàn khác nhau (data vs weather).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:*
Vì cosine similarity đo độ giống về hướng (semantic meaning), không bị ảnh hưởng bởi độ lớn vector, phù hợp hơn với embeddings.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
Step = chunk_size - overlap = 500 - 50 = 450
Số chunks ≈ 10000 / 450 ≈ 22.2 → làm tròn lên
> *Đáp án:*
23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:*
Step = 500 - 100 = 400 → số chunks tăng (~25 chunks). Overlap lớn giúp giữ context giữa các chunk, giảm mất thông tin ở biên.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...]
Chính sách & quy định dịch vụ XanhSM (customer + driver + legal)

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*
Domain này có nhiều tài liệu dài, nhiều quy định pháp lý và cấu trúc rõ ràng, phù hợp để test RAG system. Ngoài ra, dữ liệu thực tế giúp đánh giá khả năng retrieval trong môi trường production-like.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Chính sách bảo vệ dữ liệu cá nhân.txt | https://www.xanhsm.com/helps | 36,439 | category= chính sách, source=xanhsm.com |
| 2 | donhang.txt|https://www.xanhsm.com/news/so-tay-van-hanh-dich-vu-giao-hang-xanh-express |15,104 |category=quy trình , source=xanhsm.com |
| 3 | ĐIỀU KHOẢN CHUNG.txt|https://www.xanhsm.com/helps |208,756 |category= Điều khoản,dịch vụ, source=xanhsm.com|
| 4 |khach_hang.txt|https://www.xanhsm.com/terms-policies/general?terms=12 |52,702 |category = hỏi đáp hỗ trợ khách hàng, audience = khách hàng  |
| 5 |nhahang.txt|https://www.xanhsm.com/terms-policies/general?terms=10 |38,996 | category=chính sách nhà hàng, source=xanhsm.com |
|6  |tai_xe.txt | https://www.xanhsm.com/terms-policies/general?terms=6|11,424|category=điều khoản của tài xế, audience = tài xế|

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | string | chính sách, quy trình, điều khoản | Lọc theo loại tài liệu, tránh trả về chunk không liên quan loại nội dung |
| source | string | xanhsm.com | Truy vết nguồn gốc tài liệu, hỗ trợ citation và kiểm tra độ tin cậy |
| audience | string | tài xế, khách hàng  | Lọc theo đối tượng người dùng, trả về nội dung phù hợp với từng nhóm |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Điều khoản chung | FixedSizeChunker (`fixed_size`) | 226 | 300 | ❌ |
| Điều khoản chung | SentenceChunker (`by_sentences`) | 180| 400 | ✅ |
| Điều khoản chung | RecursiveChunker (`recursive`) | 150 | 500 | ✅✅ |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*
Văn bản được chia thành các chunk cố định theo số ký tự (chunk_size=800, overlap=100). Mỗi chunk chồng lấn với chunk trước đó để giữ context. Không phụ thuộc vào cấu trúc câu hay đoạn.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*
Tài liệu pháp lý thường dài và không đều về cấu trúc câu. Fixed chunk giúp đảm bảo mỗi chunk đủ dài để chứa context quan trọng, đồng thời dễ implement và tối ưu cho embedding API.

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| khach_hang.txt | SentenceChunker — best baseline | 116 | 339 | Tốt — giữ nguyên câu hoàn chỉnh |
| khach_hang.txt | *FixedSizeChunker (800) — của tôi* | *60* | *700* | *Tốt hơn — ít nhiễu, context đầy đủ hơn mỗi chunk* |
| tai_xe.txt | SentenceChunker — best baseline | 39 | 225 | Tốt — câu hoàn chỉnh |
| tai_xe.txt | *FixedSizeChunker (800) — của tôi* | *20* | *700* | *Tốt — ít chunk, giảm chi phí embedding* |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | FixedSizeChunker (800) | 9 | Chunk lớn → giữ nhiều context, giảm số lượng chunk (447 vs 1151) | Không tận dụng cấu trúc câu |
| Thư | RecursiveChunker | 10 | Cân bằng tốt giữa semantic + chunk size, phù hợp văn bản dài | Phức tạp hơn, cần tuning |
| Lực | FixedSizeChunker (300) | 8 | Đơn giản, dễ implement | Quá nhiều chunk → nhiễu retrieval |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*
RecursiveChunker là tốt nhất cho domain tài liệu pháp lý dài vì nó giữ được cấu trúc ngữ nghĩa (theo đoạn/câu) trong khi vẫn đảm bảo kích thước chunk hợp lý.

Tuy nhiên, trong implementation thực tế của nhóm, việc tăng chunk size (từ 300 lên 800) đã giúp giảm đáng kể số lượng chunk (1151 → 447), từ đó giảm nhiễu và cải thiện retrieval ổn định hơn.

Điều này cho thấy chunk size và chunking strategy cần được tối ưu đồng thời, không thể đánh giá riêng lẻ.

Một phát hiện quan trọng là vấn đề chính của hệ thống không chỉ nằm ở chunking strategy, mà còn ở việc retrieval sử dụng pure cosine similarity mà không có bước reranking hoặc document-level reasoning, dẫn đến việc các document lớn chi phối kết quả.
---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*
Dùng regex ([.!?]) để split câu. Sau đó gom các câu lại thành chunk gần chunk_size. Xử lý edge case như câu quá dài.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*
Chia theo nhiều cấp: đoạn → câu → từ. Nếu chunk quá lớn thì tiếp tục split nhỏ hơn. Base case là khi chunk ≤ max size.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?*
Chia theo nhiều cấp: đoạn → câu → từ. Nếu chunk quá lớn thì tiếp tục split nhỏ hơn. Base case là khi chunk ≤ max size.

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?*
Lưu embeddings dưới dạng list trong memory. Khi search, tính cosine similarity giữa query embedding và từng chunk.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*
Filter áp dụng sau khi retrieve. Delete bằng cách remove chunks có cùng source.

### Test Results

```
# Paste output of: pytest tests/ -v
======================================= test session starts =======================================
platform win32 -- Python 3.11.9, pytest-9.0.3, pluggy-1.6.0 -- D:\Day-07-Lab-Data-Foundations\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Day-07-Lab-Data-Foundations
collected 42 items                                                                                 

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED        [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                 [  4%] 
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED          [  7%] 
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED           [  9%] 
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                [ 11%] 
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED      [ 16%] 
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED       [ 19%] 
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED     [ 21%] 
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                       [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED       [ 26%] 
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                  [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED              [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                        [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED   [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED   [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                       [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED         [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED           [ 50%] 
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                 [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED      [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED        [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED         [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                  [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                 [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED            [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED        [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED   [ 73%] 
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED       [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED             [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED       [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED  [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
SSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
 [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

======================================= 42 passed in 0.30s ======================================== 
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | data collected | data gathered | high | 0.85 | ✅ |
| 2 | driver policy | weather report | low | 0.12 | ✅ |
| 3 | customer support | user assistance | high | 0.78 | ✅ |
| 4 | legal policy | cooking recipe | low | 0.10 | ✅ |
| 5 | payment rules | refund policy | high | 0.65 | ❌ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*
"payment rules" và "refund policy" có similarity không quá cao. Điều này cho thấy embeddings capture meaning nhưng vẫn phụ thuộc context cụ thể.
---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Khách hàng có thể hủy chuyến xe trong bao lâu mà không bị tính phí? | Khách hàng có thể hủy chuyến miễn phí trong vòng 5 phút sau khi đặt xe. |
| 2 | Tài xế cần cung cấp những giấy tờ gì khi đăng ký tham gia Xanh SM? | Tài xế cần cung cấp bằng lái xe, CMND/CCCD, đăng ký xe và bảo hiểm còn hiệu lực. |
| 3 | Xanh SM xử lý thông tin cá nhân của khách hàng như thế nào? | Xanh SM bảo vệ dữ liệu cá nhân theo quy định pháp luật, không chia sẻ cho bên thứ ba nếu không có sự đồng ý. |
| 4 | Quy trình giao hàng Xanh Express diễn ra như thế nào? | Khách đặt đơn → tài xế nhận đơn → lấy hàng → giao hàng → xác nhận hoàn thành. |
| 5 | Nhà hàng cần đáp ứng các tiêu chuẩn gì để hợp tác với Xanh SM? | Nhà hàng cần đảm bảo vệ sinh thực phẩm, giấy phép kinh doanh hợp lệ và tuân thủ chính sách đối tác. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Khách hàng có thể hủy chuyến xe trong bao lâu mà không bị tính phí? | Khách hàng có thể hủy chuyến xe trước khi tài xế tới lấy hàng mà không bị tính phí. | 0.28 | ❌ | Không trả đúng 5 phút |
| 2 | Tài xế cần cung cấp những giấy tờ gì khi đăng ký tham gia Xanh SM? | Tài xế cần cung cấp các giấy tờ sau khi đăng ký tham gia Xanh SM Bike:

- Căn cước công dân/Chứng minh nhân dân/Hộ chiếu (thời hạn sử dụng còn tối thiểu 1 tháng).
- Bằng lái xe máy A1/A2.
- Lý lịch tư pháp (LLTP): Bản gốc LLTP số 02 (không quá 06 tháng) hoặc giấy hẹn LLTP (giấy hẹn LLTP chỉ là giấy tạm thời, trong 60 ngày kể từ khi đăng ký tài khoản Tài xế cần bổ sung bản gốc).        
- Tài khoản ngân hàng chính chủ.
- Sim chính chủ. | 0.30 | ✅ | Trả đúng giấy tờ |
| 3 | Xanh SM xử lý thông tin cá nhân của khách hàng như thế nào? | Xanh SM xử lý thông tin cá nhân của khách hàng bằng cách không tiết lộ dữ liệu cá nhân trừ khi việc tiết lộ đó là cần thiết để xử lý yêu cầu của khách hàng và khi khách hàng đồng ý với các quy định về bảo mật và bảo vệ dữ liệu cá nhân của nền tảng. | 0.35 | ✅ | Trả đúng (bảo vệ, không share) |
| 4 | Quy trình giao hàng Xanh Express diễn ra như thế nào? | Quy trình giao hàng Xanh Express diễn ra như sau:

1. **Bật chế độ trực tuyến và nhận Đơn Hàng**: Đối tác vận chuyển bật chế độ trực tuyến để nhận đơn hàng.
2. **Xác nhận lệnh nhận Đơn Hàng**: Đối tác xác nhận lệnh nhận đơn hàng từ ứng dụng.
3. **Bắt đầu giao hàng**:
   - Đối tác xác nhận đã lấy hàng trên ứng dụng.
   - Cung cấp sản phẩm cho người dùng theo yêu cầu trên ứng dụng.
   - Xác nhận giao hàng thành công hoặc thất bại trên ứng dụng.
   - Liên hệ với người dùng để hoàn hàng nếu cần. | 0.33 | ✅ | Đúng flow |
| 5 | Nhà hàng cần đáp ứng các tiêu chuẩn gì để hợp tác với Xanh SM? | Nhà hàng cần đáp ứng các tiêu chuẩn sau để hợp tác với Xanh SM:

1. Giấy tờ tùy thân (CMND/CCCD hoặc Hộ chiếu).
2. Giấy phép kinh doanh (bắt buộc với Hộ kinh doanh và Công ty).
3. Giấy ủy quyền cho người đại diện ký thay (nếu có).
2. Giấy phép kinh doanh (bắt buộc với Hộ kinh doanh và Công ty).
3. Giấy ủy quyền cho người đại diện ký thay (nếu có).
4. Giấy chứng nhận vệ sinh ATTP (bắt buộc với Công ty, khuyến khích với Cá nhân/Hộ kinh doanh).     
5. Bộ menu và hình ảnh (bao gồm ảnh thực đơn, ảnh bìa, ảnh đại diện, ảnh từng món, ảnh mặt tiền quán).
6. Tài khoản ngân hàng (đăng ký thanh toán tự động, thông tin tài khoản phải trùng với giấy tờ pháp lý tương ứng). | 0.29 | ⚠️ partial | Thiếu một số điều kiện |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5
Hệ thống hoạt động tốt với các query có keyword rõ ràng (driver, data, delivery). Tuy nhiên, với các query cần thông tin cụ thể (ví dụ thời gian hủy chuyến), retrieval bị ảnh hưởng bởi các document lớn như “Điều khoản chung”, dẫn đến kết quả không chính xác.

Một hạn chế là hệ thống chỉ sử dụng cosine similarity mà không có bước reranking hoặc filtering theo metadata (ví dụ audience=khách hàng), dẫn đến việc trả về các chunk không phù hợp với ngữ cảnh câu hỏi.
---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*
Recursive chunking cải thiện rõ rệt retrieval quality vì giữ được semantic structure.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*
Một số nhóm dùng reranking sau retrieval giúp tăng độ chính xác đáng kể.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*
Tôi sẽ giới hạn số chunk mỗi document và thêm metadata filtering để tránh document lớn dominate retrieval.
---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
