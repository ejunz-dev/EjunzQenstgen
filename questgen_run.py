from pprint import pprint
from Questgen import main

# 初始化 QGen 类
qg = main.QGen()

# 输入文本
payload = {
    "input_text": """
    Sachin Tendulkar, often referred to as the 'God of Cricket', has scored a hundred centuries in international cricket.
    He was born on April 24, 1973, in Mumbai, India. Tendulkar is also the first cricketer to score a double century in an ODI.
    His career spans over two decades, earning him numerous awards, including the Bharat Ratna.
    """,
    "max_questions": 15  # 增加生成问题的上限
}

output = qg.predict_mcq(payload)
print(output)


# 调用 predict_mcq 方法
output = qg.predict_mcq(payload)

# 打印输出结果
pprint(output)
