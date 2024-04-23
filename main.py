from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from tools import pre_process

load_dotenv()


def oscar(filme, ano, llm):
    prompt = PromptTemplate(
        input_variables=['filme', 'ano'],
        template="Quantos oscars o filme {filme} ganou em {ano}"
    )

    oscar_chain = LLMChain(llm=llm, prompt=prompt)

    chain_response = oscar_chain({'filme': filme, 'ano': ano})
    return chain_response


llm = OpenAI(temperature=0.5, model='gpt-3.5-turbo-instruct')

if __name__ == "__main__":
    pre_process()
    response = oscar('Oppenheimer', 2024, llm)
    print(response['text'])
