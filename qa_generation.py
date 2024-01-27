from langchain.chains import QAGenerationChain
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI

filename = 'data/672 - Seth Godin — The Pursuit of Meaning, The Life-Changing Power of Choosing Your Attitude, Overcoming Rejection, Life Lessons from Zig Ziglar, and Committing to Making Positive Change.txt'
loader = TextLoader(filename)
raw_transcript = loader.load()[0]
txt_godin = raw_transcript.page_content

filename = 'data/661 - Dr. Peter Attia — The Science and Art of Longevity.txt'
loader = TextLoader(filename)
raw_transcript = loader.load()[0]
txt_attia = raw_transcript.page_content

filename = 'data/695 - Shane Parrish on Wisdom from Warren Buffett, Rules for Better Thinking, How to Reduce Blind Spots, The Dangers of Mental Models, and More (#695).txt'
loader = TextLoader(filename)
raw_transcript = loader.load()[0]
txt_parrish= raw_transcript.page_content

print("Generating QA chain for Godin…")
chain = QAGenerationChain.from_llm(ChatOpenAI(temperature = 0))
qa_godin = chain.run(txt_godin)

print("Generating QA chain for Attia")
chain = QAGenerationChain.from_llm(ChatOpenAI(temperature = 0))
qa_attia = chain.run(txt_attia)

print("Generating QA chain for Parrish")
chain = QAGenerationChain.from_llm(ChatOpenAI(temperature = 0))
qa_parrish= chain.run(txt_parrish)

qa_all = qa_godin + qa_attia + qa_parrish
qa_all

print("qa_all: " + ', '.join(map(str, qa_all)))
