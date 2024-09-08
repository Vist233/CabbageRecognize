import base64
import openai



image_path = “path/to/your/image.jpg”


with open(image_path, “rb”) as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode(“utf-8”)


input_data = {
    “role”: “system”,
    “content”: “请你学习第一张Strandard中四种白菜的抱合类型，之后判断出每一张图片中白菜的抱合类型。按照 图片名：饱和类型 来回答。不清楚的请回答不知道。”,
    “image”: encoded_image  
}




# 设置API密钥
openai.api_key = “YOUR_OPENAI_API_KEY”

# 发送请求
response = openai.ChatCompletion.create(
    model=”gpt-3.5-turbo”,
    messages=[input_data],
)

# 获取模型的回复
model_reply = response.choices[0].message[“content”]