import os

from summarize.settings import OPENAI_API_KEY
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("test.pdf")
pages = loader.load_and_split()
text = ' '.join([page.page_content.replace('\t', ' ') for page in pages])

import re
def clean_text(text):
   # Remove the specific phrase 'Free eBooks at Planet eBook.com' and surrounding whitespace
   cleaned_text = re.sub(r'\s*Free eBooks at Planet eBook\.com\s*', '', text, flags=re.DOTALL)
   # Remove extra spaces
   cleaned_text = re.sub(r' +', ' ', cleaned_text)
   # Remove non-printable characters, optionally preceded by 'David Copperfield'
   cleaned_text = re.sub(r'(David Copperfield )?[\x00-\x1F]', '', cleaned_text)
   # Replace newline characters with spaces
   cleaned_text = cleaned_text.replace('\n', ' ')
   # Remove spaces around hyphens
   cleaned_text = re.sub(r'\s*-\s*', '', cleaned_text)
   return cleaned_text
clean_text=clean_text(text)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
Tokens = llm.get_num_tokens(clean_text)
print (f"We have {Tokens} tokens in the book")

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
text_splitter = SemanticChunker(
   OpenAIEmbeddings(), breakpoint_threshold_type="interquartile"
)
docs = text_splitter.create_documents([clean_text])
import numpy as np
import openai
def get_embeddings(text):
   response = openai.embeddings.create(
       model="text-embedding-3-small",
       input=text
   )
   return response.data
embeddings=get_embeddings([doc.page_content for doc in docs]
)
import pandas as pd
content_list = [doc.page_content for doc in docs]
df = pd.DataFrame(content_list, columns=['page_content'])
vectors = [embedding.embedding for embedding in embeddings]
array = np.array(vectors)
embeddings_series = pd.Series(list(array))
df['embeddings'] = embeddings_series


import numpy as np
import faiss

# Convert to float32 if not already
array = array.astype('float32')

num_clusters = 50 if len(array) > 50 else len(array)

# Vectors dimensionality
dimension = array.shape[1]

# Train KMeans with Faiss
kmeans = faiss.Kmeans(dimension, num_clusters, niter=20, verbose=True)
kmeans.train(array)

# Directly access the centroids
centroids = kmeans.centroids

# Create a new index for the original dataset
index = faiss.IndexFlatL2(dimension)

# Add original dataset to the index
index.add(array)

D, I = index.search(centroids, 1)
sorted_array = np.sort(I, axis=0)
sorted_array=sorted_array.flatten()
extracted_docs = [docs[i] for i in sorted_array]
model = ChatOpenAI(temperature=0,model="gpt-4")
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
You will be given different passages from a book one by one. Provide a summary of the following text. Your result must be detailed and atleast 2 paragraphs. When summarizing, directly dive into the narrative or descriptions from the text without using introductory phrases like 'In this passage'. Directly address the main events, characters, and themes, encapsulating the essence and significant details from the text in a flowing narrative. The goal is to present a unified view of the content, continuing the story seamlessly as if the passage naturally progresses into the summary

Passage:

```{text}```
SUMMARY:
"""
)
chain= (
    prompt
   | model
   |StrOutputParser() )
from tqdm import tqdm
final_summary = ""

for doc in tqdm(extracted_docs, desc="Processing documents"):
   # Get the new summary.
   new_summary = chain.invoke({"text": doc.page_content})
   # Update the list of the last two summaries: remove the first one and add the new one at the end.
   final_summary+=new_summary

last_summary = final_summary
from fpdf import FPDF

class PDF(FPDF):
   def header(self):
       # Select Arial bold 15
       self.set_font('Arial', 'B', 15)
       # Move to the right
       self.cell(80)
       # Framed title
       self.cell(30, 10, 'Summary', 1, 0, 'C')
       # Line break
       self.ln(20)
   
   def footer(self):
       # Go to 1.5 cm from bottom
       self.set_y(-15)
       # Select Arial italic 8
       self.set_font('Arial', 'I', 8)
       # Page number
       self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')

# Instantiate PDF object and add a page
pdf = PDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# Ensure the 'last_summary' text is treated as UTF-8
# Replace 'last_summary' with your actual text variable if different
# Make sure your text is a utf-8 encoded string
last_summary_utf8 = last_summary.encode('latin-1', 'replace').decode('latin-1')
pdf.multi_cell(0, 10, last_summary_utf8)

# Save the PDF to a file
pdf_output_path = "summary.pdf"
pdf.output(pdf_output_path)

