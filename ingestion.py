from langchain.document_loaders import TextLoader
from llama_index.vector_stores import PineconeVectorStore
import pinecone
from llama_index.node_parser import SentenceSplitter
from llama_index.schema import TextNode
from llama_index.embeddings import OpenAIEmbedding
from episodes_metadata import episodes_metadata_dict_2023
import os

def get_transcript(filename):
    loader = TextLoader(filename)
    raw_transcript = loader.load()[0]
    return raw_transcript.page_content

def create_vector_store():
    # We first need to create a free Pinecone account and get an API key (1 index is free)
    api_key = os.environ["PINECONE_API_KEY"]
    environment = os.environ["PINECONE_ENVIRONMENT"]
    pinecone.init(api_key=api_key, environment=environment)
    
    index_name = "tim-gpt-index"
    
    # dimensions are for text-embedding-ada-002
    pinecone.create_index(
      index_name, dimension=1536, metric="euclidean", pod_type="p1"
    )
    
    pinecone_index = pinecone.Index(index_name)
    
    # Create PineconeVectorStore
    # Simple wrapper abstraction to use in LlamaIndex. Wrap in StorageContext so we can easily load in Nodes.
    return PineconeVectorStore(pinecone_index=pinecone_index)

vector_store = create_vector_store()

folder_path = 'transcripts_2023'

for filename in os.listdir(folder_path):
    # Skip non-text files like .DS_Store on Mac
    if not filename.endswith('.txt'):
        continue
    
    file_path = os.path.join(folder_path, filename)
    if not os.path.isfile(file_path):
        continue
    
    # we can use the filename to get the episode id
    id = filename.split(' ')[0]

    print(f"Generating embeddings for episode #{id}")

    episode_metadata = episodes_metadata_dict_2023[id]
    
    transcript = get_transcript(file_path)

    # The great thing about SentenceSplitter is that it parses text with a preference for complete sentences.
    # This prevents having a sentence split in half across two nodes.
    text_parser = SentenceSplitter(chunk_size = 1024)

    text_chunks = text_parser.split_text(transcript)

    # Manually Construct Nodes from Text Chunks
    # We can use the metadata field to store the episode id, title, and url
    # This will allow 2 things:
    # 1. The episode title provides information about the context of the text
    # 2. We can show the episode title and URL in the search results
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
            metadata={
                "episode_id": id,
                "episode_title": episode_metadata['title'],
                "episode_url": episode_metadata['url']
            }
        )
        nodes.append(node)

    # Generate Embeddings for each Node
    embed_model = OpenAIEmbedding()

    for node in nodes:
        # We also embed the metadata in the vector to store since it is useful for search results
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    vector_store.add(nodes)
