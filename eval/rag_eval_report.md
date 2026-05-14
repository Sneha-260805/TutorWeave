# EduAgent — RAG Evaluation Report

- **Generated:** 2026-05-14 13:25
- **Mode:** full
- **Questions evaluated:** 18
- **Judge model:** llama-3.1-8b-instant
- **RAG top-N:** 3
- **Judge calls skipped (rate-limit):** 0

---
## Scorecard

### Retrieval Layer

| Metric | Score | Notes |
|--------|-------|-------|
| Hit Rate @3 | **83.3%** | Exact source Q in top-3 |
| MRR | **0.8333** | Mean Reciprocal Rank |
| Topic Precision | **83.3%** | Retrieved docs match correct topic |
| Level Precision | **100.0%** | Retrieved docs match correct level |
| Avg Semantic Sim | **0.7153** | Cosine sim query vs retrieved Qs |
| Level Detection Accuracy | **77.8%** | Correctly detected beginner/intermediate/advanced |

### Generation Layer  (LLM-as-Judge, 1–5)

| Metric | Avg | What it measures |
|--------|-----|-----------------|
| Faithfulness | **5.00** | Answer grounded in context (not hallucinated) |
| Answer Relevancy | **5.00** | Answer addresses the student's question |
| Context Utilization | **4.44** | Tutor built upon retrieved examples |
| Level Appropriateness | **4.94** | Explanation pitched at the right difficulty |

---
## Per-Question Results

### Q1. How does the intrinsic dimensionality of feature maps evolve across very deep co…
- **True:** `advanced` / `convolutional neural networks`
- **Detected:** `advanced` (✓) / `convolutional neural networks`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.734  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=5/5  level=5/5
  > The tutor's response accurately and comprehensively addresses the question, builds upon the retrieved context, and uses language suitable for an advanced student.

### Q2. What are the theoretical limitations of GRUs in recognizing or generating sequen…
- **True:** `advanced` / `recurrent neural networks`
- **Detected:** `advanced` (✓) / `recurrent neural networks`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.699  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=5/5  level=5/5
  > The tutor's response accurately and thoroughly addresses the question, building upon the retrieved context and using language suitable for an advanced student.

### Q3. In what specific scenarios can a probabilistic framework, such as a Hidden Marko…
- **True:** `advanced` / `natural language processing`
- **Detected:** `advanced` (✓) / `natural language processing`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.825  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=4/5  level=5/5
  > The tutor's response accurately built upon the provided context and addressed the question with suitable depth for an advanced student.

### Q4. From an information-theoretic perspective, how does the redundancy inherent in o…
- **True:** `advanced` / `transfer learning`
- **Detected:** `advanced` (✓) / `overfitting and regularization`
- **Retrieval:** hit=NO  mrr=0.000  sim=0.446  topic_prec=0%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=5/5  level=5/5
  > The tutor's response accurately and comprehensively addresses the question, effectively building upon the retrieved context and using language suitable for an advanced student.

### Q5. What are the fundamental limitations in developing universally applicable, fully…
- **True:** `advanced` / `ethical AI and bias`
- **Detected:** `advanced` (✓) / `ethical AI and bias`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.845  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=5/5  level=5/5
  > The tutor's response accurately and comprehensively addresses the question, builds upon the retrieved context, and uses language suitable for an advanced student.

### Q6. How do concept drift and data distribution shifts in the underlying embedding sp…
- **True:** `advanced` / `vector databases`
- **Detected:** `advanced` (✓) / `machine learning basics`
- **Retrieval:** hit=NO  mrr=0.000  sim=0.556  topic_prec=0%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=4/5  level=5/5
  > The tutor's response accurately builds upon the provided context and addresses the question at an appropriate level for an advanced student.

### Q7. Why is a Convolutional Neural Network's architecture particularly well-suited fo…
- **True:** `beginner` / `convolutional neural networks`
- **Detected:** `advanced` (✗ MISMATCH) / `convolutional neural networks`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.708  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=4/5  level=5/5
  > The tutor's response accurately and thoroughly addresses the question, effectively building upon the retrieved context, and is well-suited for an advanced student.

### Q8. Does a GRU completely erase its memory of previous steps when it processes new i…
- **True:** `beginner` / `recurrent neural networks`
- **Detected:** `beginner` (✓) / `recurrent neural networks`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.825  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=5/5  level=5/5
  > The tutor's response accurately and thoroughly addresses the question, builds upon the retrieved context, and uses language suitable for a beginner student.

### Q9. What is the most basic step tokenization takes to prepare a sentence like "Hello…
- **True:** `beginner` / `natural language processing`
- **Detected:** `beginner` (✓) / `machine learning basics`
- **Retrieval:** hit=NO  mrr=0.000  sim=0.362  topic_prec=0%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=4/5  level=5/5
  > The tutor's answer is well-supported by the context, directly addresses the question, and uses the context to explain the concept, but could have more explicitly referenced the examples.

### Q10. How is starting a new image recognition project with a pre-trained model's featu…
- **True:** `beginner` / `transfer learning`
- **Detected:** `advanced` (✗ MISMATCH) / `transfer learning`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.739  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=5/5  level=5/5
  > The tutor's response accurately and thoroughly addresses the question, effectively utilizing the provided context and presenting complex concepts in a suitable manner for an advanced student.

### Q11. Is responsible AI just about making sure the AI works perfectly without any mist…
- **True:** `beginner` / `ethical AI and bias`
- **Detected:** `beginner` (✓) / `ethical AI and bias`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.856  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=4/5  level=5/5
  > The tutor's response accurately and thoroughly addresses the question, effectively incorporates the retrieved context, and uses language suitable for a beginner student.

### Q12. Why is similarity search a powerful tool for recommending movies or suggesting r…
- **True:** `beginner` / `vector databases`
- **Detected:** `beginner` (✓) / `vector databases`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.732  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=4/5  level=5/5
  > The tutor's response accurately and thoroughly addresses the question, effectively utilizes the provided context, and is written at a suitable level for a beginner student.

### Q13. How should the spatial resolution of input images guide the selection of kernel …
- **True:** `intermediate` / `convolutional neural networks`
- **Detected:** `intermediate` (✓) / `convolutional neural networks`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.831  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=4/5  level=5/5
  > The tutor's response accurately and thoroughly addresses the question, effectively utilizes the provided context, and is well-suited for an intermediate student, but could benefit from a more explicit connection to the second context example.

### Q14. What are the primary trade-offs between increasing the number of LSTM layers ver…
- **True:** `intermediate` / `recurrent neural networks`
- **Detected:** `advanced` (✗ MISMATCH) / `recurrent neural networks`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.796  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=4/5  level=5/5
  > The tutor's response accurately and thoroughly addresses the question, effectively building upon the retrieved context, and is well-suited for an advanced student.

### Q15. A machine translation model produces grammatically incorrect output for compound…
- **True:** `intermediate` / `natural language processing`
- **Detected:** `intermediate` (✓) / `natural language processing`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.675  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=4/5  level=5/5
  > The tutor effectively referenced the retrieved examples and provided a clear explanation suitable for an intermediate student.

### Q16. When should one prioritize a shallow, feature-alignment approach over a deep, en…
- **True:** `intermediate` / `transfer learning`
- **Detected:** `advanced` (✗ MISMATCH) / `transfer learning`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.728  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=5/5  level=5/5
  > The tutor's response accurately and thoroughly addresses the question, leveraging the provided context without introducing any extraneous information.

### Q17. Why might simply balancing protected attributes in a training dataset not fully …
- **True:** `intermediate` / `ethical AI and bias`
- **Detected:** `intermediate` (✓) / `ethical AI and bias`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.760  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=5/5  level=5/5
  > The tutor's response accurately and thoroughly addresses the question, building upon the provided context without introducing any new, unsupported information.

### Q18. How does the choice of distance metric (e.g., Euclidean vs. Cosine) influence th…
- **True:** `intermediate` / `vector databases`
- **Detected:** `intermediate` (✓) / `vector databases`
- **Retrieval:** hit=YES  mrr=1.000  sim=0.759  topic_prec=100%  level_prec=100%
- **Judge:** faith=5/5  relev=5/5  ctx=4/5  level=4/5
  > The tutor's response accurately and thoroughly addresses the question, effectively utilizes the retrieved context, and is suitable for an intermediate student, but could benefit from a more explicit connection to the examples.

---
## Weaknesses & Recommendations

**Retrieval misses (3 questions not found in top-3):**
Fix: Increase RAG_TOP_N, or improve topic detection.
- From an information-theoretic perspective, how does the redundancy inherent in over-parame
- How do concept drift and data distribution shifts in the underlying embedding space impact
- What is the most basic step tokenization takes to prepare a sentence like "Hello, world!" 
