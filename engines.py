"""
Saare heavy OCR engines ka wrapper.
Lazy loading + Streamlit caching use kiya gaya hai.
"""

import numpy as np


# ==========================================================
# 1. TESSERACT
# ==========================================================
def run_tesseract(image, lang="en"):
    import pytesseract
    tess_lang = {
        "en": "eng",
        "hi": "hin",
        "en+hi": "eng+hin",
    }.get(lang, "eng")
    return pytesseract.image_to_string(image, lang=tess_lang)


# ==========================================================
# 2. EASYOCR
# ==========================================================
_easy_cache = {}

def run_easyocr(img_np, lang="en", gpu=False):
    import easyocr
    langs = ["en"] if lang == "en" else (["hi"] if lang == "hi" else ["en", "hi"])
    key = tuple(langs) + (gpu,)
    if key not in _easy_cache:
        _easy_cache[key] = easyocr.Reader(list(langs), gpu=gpu)
    reader = _easy_cache[key]
    result = reader.readtext(img_np, detail=0, paragraph=True)
    return "\n".join(result)


# ==========================================================
# 3. PADDLEOCR
# ==========================================================
_paddle_cache = {}

def run_paddleocr(img_np, lang="en", gpu=False):
    from paddleocr import PaddleOCR
    key = (lang, gpu)
    if key not in _paddle_cache:
        _paddle_cache[key] = PaddleOCR(
            use_angle_cls=True,
            lang="en" if lang != "hi" else "hi",
            use_gpu=gpu,
            show_log=False,
        )
    ocr = _paddle_cache[key]
    result = ocr.ocr(img_np, cls=True)
    lines = []
    if result and result[0]:
        for line in result[0]:
            lines.append(line[1][0])
    return "\n".join(lines)


# ==========================================================
# 4. RAPIDOCR (ONNX - super fast)
# ==========================================================
_rapid = None

def run_rapidocr(img_np):
    global _rapid
    if _rapid is None:
        from rapidocr_onnxruntime import RapidOCR
        _rapid = RapidOCR()
    result, _ = _rapid(img_np)
    if not result:
        return ""
    return "\n".join([r[1] for r in result])


# ==========================================================
# 5. DOCTR (Document Text Recognition)
# ==========================================================
_doctr = None

def run_doctr(img_np):
    global _doctr
    if _doctr is None:
        from doctr.models import ocr_predictor
        _doctr = ocr_predictor(pretrained=True)
    result = _doctr([img_np])
    lines = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                lines.append(" ".join([w.value for w in line.words]))
    return "\n".join(lines)


# ==========================================================
# 6. KERAS-OCR
# ==========================================================
_keras = None

def run_kerasocr(img_np):
    global _keras
    if _keras is None:
        import keras_ocr
        _keras = keras_ocr.pipeline.Pipeline()
    prediction = _keras.recognize([img_np])
    return "\n".join([w[0] for w in prediction[0]])


# ==========================================================
# 7. SURYA OCR (heavy, layout-aware, multilingual)
# ==========================================================
_surya_models = None

def run_surya(image, lang="en"):
    global _surya_models
    from surya.ocr import run_ocr
    from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor

    if _surya_models is None:
        _surya_models = {
            "det_model": load_det_model(),
            "det_processor": load_det_processor(),
            "rec_model": load_rec_model(),
            "rec_processor": load_rec_processor(),
        }

    langs = [["en"]] if lang == "en" else ([["hi"]] if lang == "hi" else [["en", "hi"]])
    predictions = run_ocr(
        [image],
        [langs[0]],
        _surya_models["det_model"],
        _surya_models["det_processor"],
        _surya_models["rec_model"],
        _surya_models["rec_processor"],
    )
    lines = []
    for pred in predictions:
        for line in pred.text_lines:
            lines.append(line.text)
    return "\n".join(lines)


# ==========================================================
# 8. TrOCR (Microsoft, transformer-based)
# ==========================================================
_trocr = None

def run_trocr(image):
    global _trocr
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    if _trocr is None:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        _trocr = (processor, model)
    processor, model = _trocr
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=512)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


# ==========================================================
# 9. MMOCR (OpenMMLab)
# ==========================================================
_mmocr = None

def run_mmocr(img_np):
    global _mmocr
    if _mmocr is None:
        from mmocr.apis import MMOCRInferencer
        _mmocr = MMOCRInferencer(det="dbnetpp", rec="svtr-small")
    result = _mmocr(img_np, return_vis=False)
    texts = result.get("predictions", [{}])[0].get("rec_texts", [])
    return "\n".join(texts)


# ==========================================================
# Registry
# ==========================================================
ENGINES = {
    "Tesseract":    {"fn": run_tesseract, "needs": "pil"},
    "EasyOCR":      {"fn": run_easyocr,   "needs": "np"},
    "PaddleOCR":    {"fn": run_paddleocr, "needs": "np"},
    "RapidOCR":     {"fn": run_rapidocr,  "needs": "np"},
    "DocTR":        {"fn": run_doctr,     "needs": "np"},
    "KerasOCR":     {"fn": run_kerasocr,  "needs": "np"},
    "SuryaOCR":     {"fn": run_surya,     "needs": "pil"},
    "TrOCR":        {"fn": run_trocr,     "needs": "pil"},
    "MMOCR":        {"fn": run_mmocr,     "needs": "np"},
}
