import os
import re
import numpy as np
import pandas as pd
import openai
import faiss
from fpdf import FPDF
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from summarize.settings import OPENAI_API_KEY


class PDFSummaryGenerator:
    def __init__(self, pdf_path, api_key):
        self.loader = PyPDFLoader(pdf_path)
        self.pages = self.loader.load_and_split()
        self.api_key = api_key

    def _clean_text(self, text):
        text = re.sub(
            r"\s*Free eBooks at Planet eBook\.com\s*", "", text, flags=re.DOTALL
        )
        text = re.sub(r" +", " ", text)
        text = re.sub(r"(David Copperfield )?[\x00-\x1F]", "", text)
        text = text.replace("\n", " ")
        text = re.sub(r"\s*-\s*", "", text)
        return text

    def _generate_clean_text(self):
        return " ".join(
            [
                self._clean_text(page.page_content.replace("\t", " "))
                for page in self.pages
            ]
        )

    def _get_embeddings(self, texts):
        response = openai.embeddings.create(model="text-embedding-3-small", input=texts)
        return response.data

    def _create_embeddings_dataframe(self, docs):
        content_list = [doc.page_content for doc in docs]
        df = pd.DataFrame(content_list, columns=["page_content"])
        vectors = [
            embedding.embedding for embedding in self._get_embeddings(content_list)
        ]
        df["embeddings"] = pd.Series(list(np.array(vectors)))
        return df

    def _train_kmeans(self, array):
        array = array.astype("float32")
        num_clusters = 50 if len(array) > 50 else len(array)
        dimension = array.shape[1]
        kmeans = faiss.Kmeans(dimension, num_clusters, niter=20, verbose=True)
        kmeans.train(array)
        return kmeans.centroids, dimension

    def _index_and_sort_documents(self, centroids, dimension, array, docs):
        index = faiss.IndexFlatL2(dimension)
        index.add(array)
        D, I = index.search(centroids, 1)
        sorted_indices = np.sort(I, axis=0).flatten()
        return [docs[i] for i in sorted_indices]

    def _generate_summary(self, docs):
        model = ChatOpenAI(temperature=0, model="gpt-4")
        prompt = ChatPromptTemplate.from_template(
            """
            You will be given different passages from a book one by one. Provide a summary of the following text. 
            Your result must be detailed and at least 2 paragraphs. When summarizing, directly dive into the narrative 
            or descriptions from the text without using introductory phrases like 'In this passage'. Directly address 
            the main events, characters, and themes, encapsulating the essence and significant details from the text 
            in a flowing narrative. The goal is to present a unified view of the content, continuing the story seamlessly 
            as if the passage naturally progresses into the summary.

            Passage:

            ```{text}```
            SUMMARY:
        """
        )
        chain = prompt | model | StrOutputParser()
        final_summary = ""
        for doc in tqdm(docs, desc="Processing documents"):
            final_summary += chain.invoke({"text": doc.page_content})
        return final_summary

    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 15)
            self.cell(80)
            self.cell(30, 10, "Summary", 1, 0, "C")
            self.ln(20)

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def generate_pdf_summary(self, output_file_name="summary.pdf"):
        clean_text = self._generate_clean_text()

        llm = ChatOpenAI(model="gpt-4o")
        tokens = llm.get_num_tokens(clean_text)
        print(f"We have {tokens} tokens in the book")

        text_splitter = SemanticChunker(
            OpenAIEmbeddings(), breakpoint_threshold_type="interquartile"
        )
        docs = text_splitter.create_documents([clean_text])

        df = self._create_embeddings_dataframe(docs)
        array = np.vstack(df["embeddings"])

        centroids, dimension = self._train_kmeans(array)
        sorted_docs = self._index_and_sort_documents(centroids, dimension, array, docs)

        last_summary = self._generate_summary(sorted_docs)

        pdf = self.PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        last_summary_utf8 = last_summary.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 10, last_summary_utf8)
        pdf.output(output_file_name)


if __name__ == "__main__":
    generator = PDFSummaryGenerator(pdf_path="test.pdf", api_key=OPENAI_API_KEY)
    generator.generate_pdf_summary("summary.pdf")
