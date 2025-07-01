from typing import List,Dict
from pathlib import Path
import re
import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader 

from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_medical_text(text: str) -> List[Dict[str, str]]:
    chapters = re.split(r"\n([IVXLCDM]{1,6}\s+[^\n]+)", text)
    structured = []

    i = 1
    while i < len(chapters):
        chapter_title = chapters[i].strip()
        chapter_content = chapters[i + 1]

        sections = re.split(r"\n(\d+\s+[^\n]+)", chapter_content)

        if len(sections) == 1:
            structured.append({
                "chapter": chapter_title,
                "section": "",
                "content": chapter_content.strip()
            })
        else:
            j = 1
            while j < len(sections):
                section_title = sections[j].strip()
                section_content = sections[j + 1].strip()
                structured.append({
                    "chapter": chapter_title,
                    "section": section_title,
                    "content": section_content
                })
                j += 2
        i += 2
    return structured



def load_and_split_pdf(pdf_path: str, parent_retriver = True ) -> List:
    """
    Load a PDF file and split it into chunks.
    
    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List: List of document chunks
    """
    chunk_size = int(os.getenv("chunk_size","500"))
    chunk_overlap = int(os.getenv("chunk_overlap","50"))
    try:
        # Convert string path to Path object
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        if not pages:
            raise ValueError(f"No content found in PDF: {pdf_path}")
        
        # Split text into chunks
        
        text_splitter = RecursiveCharacterTextSplitter(
            separators = [
                r"\n[IVXLCDM]{1,6}\s+[^\n]+",  # I, II, III 
                r"\n\d+\s+[^\n]+",             # 1 little title
                r"\n\n",                       # break 
                r"\n",                         # new line 
            ],
            chunk_size=chunk_size,             
            chunk_overlap=chunk_overlap,           
            length_function=len,         
            is_separator_regex=True      
            )
        chunks = text_splitter.split_documents(pages) 
        # Add unique id to each chunk's metadata
        total_chunks= []
        for idx, chunk in enumerate(chunks):
            if len(chunk.page_content) < chunk_overlap and chunk.metadata["page"] + 1 == chunks[idx+1].metadata["page"]: 
                continue
            # Use a combination of source, page, and chunk index for uniqueness if available
            source = chunk.metadata.get("source", str(pdf_path))
            page = chunk.metadata.get("page", 0)
            chunk.metadata["id"] = f"{source}:{page}:{idx}"
            total_chunks.append(chunk) 
        for chunk in total_chunks:    
            print(chunk.metadata)
        return total_chunks
        
    except Exception as e:
        raise Exception(f"Error processing PDF {pdf_path}: {str(e)}") 