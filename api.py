from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from Questgen.main import QGen, BoolQGen, AnswerPredictor
import uvicorn
from fastapi import Request

app = FastAPI(title="Questgen API Service")

qgen = QGen()
bool_qgen = BoolQGen()
answer_predictor = AnswerPredictor()

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

@app.post("/generate-mcq", response_model=Dict)
async def generate_mcq(request: Request):
    try:
        raw_body = await request.json()
        print("Raw request body:", raw_body)  # 打印接收到的原始请求体

        # 提取并处理 input_text
        input_text_data = raw_body.get("input_text")
        max_questions_data = raw_body.get("max_questions")

        # 处理 input_text
        if isinstance(input_text_data, dict):
            input_text = input_text_data.get("input_text")
        elif isinstance(input_text_data, str):
            input_text = input_text_data
        else:
            raise ValueError(f"Invalid input_text: {input_text_data} (expected string or dict)")

        # 处理 max_questions
        if isinstance(max_questions_data, int):
            max_questions = max_questions_data
        elif isinstance(max_questions_data, str):
            try:
                max_questions = int(max_questions_data)
            except ValueError:
                raise ValueError(f"Invalid max_questions: {max_questions_data} (expected integer)")
        else:
            raise ValueError(f"Invalid max_questions: {max_questions_data} (expected integer or string)")

        # 验证提取的值
        if not input_text or not isinstance(input_text, str):
            raise ValueError(f"Invalid input_text after extraction: {input_text}")
        if not isinstance(max_questions, int) or max_questions <= 0:
            raise ValueError(f"Invalid max_questions after extraction: {max_questions}")

        # 调用模型
        payload = {"input_text": input_text, "max_questions": max_questions}
        print("Payload sent to model:", payload) 
        result = qgen.predict_mcq(payload)
        print("Prediction result:", result) 
        return result
    except Exception as e:
        print(f"Error occurred: {e}")
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10086)
