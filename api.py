from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from Questgen.main import QGen, BoolQGen, AnswerPredictor
import uvicorn
import random
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
        print("Raw request body:", raw_body)  

        input_text_data = raw_body.get("input_text")
        max_questions_data = raw_body.get("max_questions")

        if isinstance(input_text_data, dict):
            input_text = input_text_data.get("input_text")
        elif isinstance(input_text_data, str):
            input_text = input_text_data
        else:
            raise ValueError(f"Invalid input_text: {input_text_data} (expected string or dict)")

        if isinstance(max_questions_data, int):
            max_questions = max_questions_data
        elif isinstance(max_questions_data, str):
            try:
                max_questions = int(max_questions_data)
            except ValueError:
                raise ValueError(f"Invalid max_questions: {max_questions_data} (expected integer)")
        else:
            raise ValueError(f"Invalid max_questions: {max_questions_data} (expected integer or string)")

        if not input_text or not isinstance(input_text, str):
            raise ValueError(f"Invalid input_text after extraction: {input_text}")
        if not isinstance(max_questions, int) or max_questions <= 0:
            raise ValueError(f"Invalid max_questions after extraction: {max_questions}")

        payload = {"input_text": input_text, "max_questions": max_questions}
        print("Payload sent to model:", payload) 
        result = qgen.predict_mcq(payload)
        print("Prediction result:", result) 
        
        for question in result.get("questions", []):
            if "options" in question and isinstance(question["options"], list):
                # 确保选项列表包含正确答案
                options = question["options"]
                correct_answer = question["answer"]
                
                # 随机插入正确答案
                if correct_answer not in options:
                    insert_index = random.randint(0, len(options))
                    options.insert(insert_index, correct_answer)
                else:
                    insert_index = options.index(correct_answer)

                # 为选项添加标签
                question["labeled_options"] = [
                    {"label": chr(65 + idx), "value": option, "is_correct": (idx == insert_index)}
                    for idx, option in enumerate(options)
                ]

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
