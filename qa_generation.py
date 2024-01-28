from langchain.chains import QAGenerationChain
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI

filename = 'data/661 - Dr. Peter Attia — The Science and Art of Longevity.txt'
loader = TextLoader(filename)
raw_transcript = loader.load()[0]
txt_attia = raw_transcript.page_content

print("Generating QA chain for Attia")
chain = QAGenerationChain.from_llm(ChatOpenAI(temperature = 0))
qa_attia = chain.run(txt_attia)

print("qa_attia: " + ', '.join(map(str, qa_attia)))
