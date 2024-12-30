from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Optional
from Questgen.main import QGen, BoolQGen, AnswerPredictor
import uvicorn
import random
from fastapi import Request
from fastapi.responses import JSONResponse
import json

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
        # 从请求体解析参数
        raw_body = await request.json()
        input_text = raw_body.get("input_text")
        max_questions = int(raw_body.get("max_questions", 0))
        domain_id = raw_body.get("domainId", "system") 
        user_id = raw_body.get("userId") 

        # 参数验证
        if not input_text or max_questions <= 0:
            raise ValueError("Invalid input_text or max_questions.")
        if not domain_id:
            raise ValueError("Invalid domainId. It cannot be empty.")

        print(f"Received domainId: {domain_id}, userId: {user_id}")
        print("Payload sent to model:", {"input_text": input_text, "max_questions": max_questions})

        payload = {"input_text": input_text, "max_questions": max_questions}
        result = qgen.predict_mcq(payload)

        # 格式化结果为合法 JSON
        questions = []
        for question in result.get("questions", []):
            options = question["options"]
            correct_answer = question["answer"]

            if correct_answer not in options:
                options.insert(random.randint(0, len(options)), correct_answer)
            labeled_options = [
                {"label": chr(65 + idx), "value": opt} for idx, opt in enumerate(options)
            ]

            questions.append({
                "question_statement": question["question_statement"],
                "labeled_options": labeled_options,
                "answer": correct_answer,
            })

        response = {
            "domainId": domain_id,  
            "userId": user_id,     
            "questions": questions
        }

        serialized_response = json.dumps(response, indent=4)
        print("Serialized Response JSON:\n", serialized_response)

        return JSONResponse(content=response)

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
    uvicorn.run(app, host="0.0.0.0", port=10001)
