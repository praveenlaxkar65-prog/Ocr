# 🔥 Heavy Multi-Engine OCR Tester

Streamlit app jo ek hi image pe **9 OCR engines** chalata hai aur compare karta hai.

## Engines
| Engine | Type | Speed | Multilingual |
|---|---|---|---|
| Tesseract | Classic | ⚡⚡⚡ | ✅ |
| EasyOCR | PyTorch | ⚡⚡ | ✅ |
| PaddleOCR | PaddlePaddle | ⚡⚡ | ✅ |
| RapidOCR | ONNX | ⚡⚡⚡ | ✅ |
| DocTR | PyTorch | ⚡⚡ | ⚠️ |
| KerasOCR | TF | ⚡ | ⚠️ |
| SuryaOCR | PyTorch | ⚡ | ✅✅ |
| TrOCR | Transformer | ⚡ | ❌ (printed only) |
| MMOCR | OpenMMLab | ⚡⚡ | ⚠️ |

## Local Setup

### Linux (recommended for heavy)
```bash
sudo apt install tesseract-ocr tesseract-ocr-hin libgl1 libglib2.0-0
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
