import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def main():
    client = OpenAI(api_key = OPENAI_API_KEY)
 
    content = input("Please tell me what you would like to ask N.E.O\n")

    print(content)
    
    

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are N.E.O., a multi-functional AI which will help me with task scheduling, search things I request for on the web, and many other features, similar to J.A.R.V.I.S from Iron Man"},
            {"role": "user", "content": "Hello, who are you?"}
        ]
    )

    print(completion)

    return


if __name__ =="__main__":
    main()