from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import VectorDBQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from qa_pairs import qa_pairs_attia
from langchain.evaluation.qa import QAEvalChain
import pandas as pd

# These 5 chunk sizes should be enough to get a good idea of the ideal size
CHUNK_SIZES_TO_EVAL = [250, 500, 1000, 2000, 3000]

# In theory we could also evalute various overlaps, but we will keep it simple for now
OVERLAP = 50

def make_splits(chunk_size, overlap, text):
    # Chunking strategy: fixed size with overlap between the chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap = overlap)
    
    # This simply gives me a list of text chunks
    splits = text_splitter.split_text(text)
    
    # We embed each text chunk into a vector embedding using OpenAI and store it in a FAISS vector store
    # Note that this this is only stored in memory, so it will not persist when the script finishes
    # We will create a new FAISS instance for each chunk size
    db = FAISS.from_texts(splits, OpenAIEmbeddings())
    
    # We create a chain that will take a question as input and return the most similar text chunk.
    # VectorDBQA stands for Vector DataBase Question Answering.
    # Temperature 0 means we are not adding any randomness to the model, we want to be as strict as possible.
    # chain_type "stuff" means uses ALL of the text from the documents in the prompt
    # Other chain types are: map_reduce, refine, map-rerank
    chain = VectorDBQA.from_chain_type(llm=ChatOpenAI(temperature=0), 
                                       chain_type="stuff", 
                                       vectorstore=db, 
                                       input_key="question")
    
    return chain

# In the context of Langchain, a "chain" is a sequence of transformations applied to the input data.
# Each transformation in the chain is a step that processes the data in some way.
# The output of one step is used as the input to the next step, forming a "chain" of transformations.
# Think about it like a pipeline, where the output of one step is the input to the next step.

# This is how a chain looks like when we print it:
# 250: VectorDBQA(
#     combine_documents_chain = StuffDocumentsChain(
#       llm_chain = LLMChain(
#         prompt = ChatPromptTemplate(
#           input_variables = ['context', 'question'],
#           messages = [
#             SystemMessagePromptTemplate(
#               prompt = PromptTemplate(
#                 input_variables=['context'],
#                 template="Use the following pieces of context to answer the user's question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{context}"
#               )
#             ),
#             HumanMessagePromptTemplate(
#               prompt = PromptTemplate(
#                 input_variables=['question'],
#                 template='{question}'
#               )
#             )
#           ]
#         ),
#         llm = ChatOpenAI(
#           client = <openai.resources.chat.completions.Completions object at 0x15d995e90>,
#           async_client = <openai.resources.chat.completions.AsyncCompletions object at 0x1696e9b10>,
#           temperature = 0.0,
#           openai_api_key = '<API_KEY>',
#           openai_proxy=''
#         )
#       ),
#       document_variable_name='context'
#     ),
#     input_key='question',
#     vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x15d988090>
#   ),

def get_transcript(filename):
    loader = TextLoader(filename)
    raw_transcript = loader.load()[0]
    return raw_transcript.page_content

transcript_attia = get_transcript('data/661 - Dr. Peter Attia — The Science and Art of Longevity.txt')
  
chains = {}
for chunk_size in CHUNK_SIZES_TO_EVAL:
    chain = make_splits(chunk_size, OVERLAP, transcript_attia)
    chains[chunk_size] = chain

def run_eval(chain):
    # the predictions are what we get from the QA chain 
    predictions = []
    # the predicted dataset contains the actual answers to the questions
    predicted_dataset = []
    for qa_pair in qa_pairs_attia:
        qa_pair_obj = {
            "question": qa_pair["question"],
            "answer": qa_pair["answer"]
        }
        predictions.append(chain(qa_pair_obj))
        predicted_dataset.append(qa_pair_obj)
    return predictions, predicted_dataset

scores_list = []
eval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(temperature=0))
for chunk_size in CHUNK_SIZES_TO_EVAL:
    print("Evaluation chunk size: ", chunk_size)
    predictions, predicted_dataset = run_eval(chains[chunk_size])
    # We evaluate the predictions against the actual answers
    graded_outputs = eval_chain.evaluate(
        predicted_dataset,
        predictions,
        question_key = "question",
        prediction_key = "result"
    )
    scores_list.append(graded_outputs)

# Here in scores_list, we have a 2 dimensional list of dictionaries
# With the key 'results' and the value 'CORRECT' or 'INCORRECT'
# A given row is for a given chunk size, a given column is for a question of the QA pairs
# e.g.
#                | question 1             | question 2               | ...
# chunk_size 250 | {'results': 'CORRECT'} | {'results': 'CORRECT'}   | ...
# chunk_size 500 | {'results': 'CORRECT'} | {'results': 'INCORRECT'} | ...
# ...
# Once we have that, we just transform the data into a dataframe and calculate the percentage of incorrect answers

stor = pd.DataFrame()
for i, chunk_size in enumerate(CHUNK_SIZES_TO_EVAL):
    d = scores_list[i]
    incorrect_counts = []
    for dictionary in d:
        if dictionary['results'] == 'INCORRECT':
            incorrect_counts.append(1)
        else:
            incorrect_counts.append(0)
    stor.loc[chunk_size,'num_incorrect']=sum(incorrect_counts)
    
qa_pair_count = len(scores_list[0])

stor['pct_incorrect'] = stor['num_incorrect'] / qa_pair_count * 100
stor['pct_correct'] = 100 - stor['pct_incorrect']

print(stor)
#       num_incorrect  pct_incorrect  pct_correct
# 250             3.0           3.75        96.25
# 500             1.0           1.25        98.75
# 1000            0.0           0.00       100.00
# 2000            1.0           1.25        98.75
# 3000            2.0           2.50        97.50

# We can see that the best chunk size is 1000, with 0% incorrect answers.
# Even though all chunking sizes have very good performance.
# To go further, we could:
# - apply the same kind of evaluation to other transcripts on various topics
# - find ways to generate more challenging questions
