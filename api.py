from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from Questgen.main import QGen, BoolQGen, AnswerPredictor
import uvicorn


# 初始化 FastAPI 应用
app = FastAPI(title="Questgen API Service")

# 初始化 Questgen 模型
qgen = QGen()
bool_qgen = BoolQGen()
answer_predictor = AnswerPredictor()

# 定义请求和响应数据结构
class MCQRequest(BaseModel):
    input_text: str
    max_questions: Optional[int] = 5

class BoolQRequest(BaseModel):
    input_text: str
    max_questions: Optional[int] = 5

class ParaphraseRequest(BaseModel):
    input_text: str
    max_questions: Optional[int] = 3

class AnswerRequest(BaseModel):
    input_text: str
    input_question: List[str]

# 定义 API 路由
@app.post("/generate-mcq", response_model=Dict)
def generate_mcq(request: MCQRequest):
    try:
        payload = {
            "input_text": request.input_text,
            "max_questions": request.max_questions,
        }
        result = qgen.predict_mcq(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-boolq", response_model=Dict)
def generate_boolq(request: BoolQRequest):
    try:
        payload = {
            "input_text": request.input_text,
            "max_questions": request.max_questions,
        }
        result = bool_qgen.predict_boolq(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/paraphrase", response_model=Dict)
def paraphrase(request: ParaphraseRequest):
    try:
        payload = {
            "input_text": request.input_text,
            "max_questions": request.max_questions,
        }
        result = qgen.paraphrase(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/answer-predict", response_model=List[str])
def answer_predict(request: AnswerRequest):
    try:
        payload = {
            "input_text": request.input_text,
            "input_question": request.input_question,
        }
        result = answer_predictor.predict_answer(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
