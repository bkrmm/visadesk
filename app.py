from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from functools import lru_cache
import uvicorn

# --- LangChain components ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# FastAPI app initialization
app = FastAPI(
    title="Visa Desk RAG API",
    description="API for querying visa-related information using RAG with Gemini",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
GOOGLE_API_KEY = "AIzaSyDCnAFTq5tS3rCrYb7M5jP90IuvitcgFLQ"

# Request/Response models
class QueryRequest(BaseModel):
    """Request model for the query endpoint."""
    question: Optional[str] = None
    data: Optional[Dict] = {}
    history: Optional[List] = []

    class Config:
        extra = "allow"  # Allow extra fields in the input JSON
        json_schema_extra = {
            "example": {
                "data": {
                    "age": 26,
                    "education": {
                        "highest_degree": "BCA",
                        "field": "Computer Science",
                        "year": 2023
                    }
                },
                "question": "Am I eligible for Canadian immigration?",
                "history": []
            }
        }


class QueryResponse(BaseModel):
    """Response model for the query endpoint."""
    answer: str
    sources: Optional[List[str]] = None


# =============================
# Load or create FAISS vector store
# =============================
@lru_cache()
def load_vector_store():
    """Load documents and create vector store with caching."""
    loader = DirectoryLoader("data", glob="*.md", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    modl_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=modl_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# Initialize components on startup
vectorstore = load_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =============================
# Gemini Model
# =============================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# =============================
# Prompt + Runnable Chain (Modern LangChain API)
# =============================
systemprompt = ChatPromptTemplate.from_template("""
You are a Canadian immigration consultant. Analyze the applicant's profile details and match them with Canadian immigration requirements to provide specific, personalized advice.

If applicant data is provided, follow these steps:
1. Analyze the applicant's:
   - Age, education, and work experience
   - Language scores (IELTS/other tests)
   - Job offer status and preferred province
   - Available funds
2. Match these details with eligible immigration programs
3. Recommend the best pathway based on their profile
4. List any additional steps or improvements needed

Format your response as follows:
1. Profile Summary (brief overview of key points)
2. Eligible Programs (in order of best match)
3. Recommended Next Steps (bullet points)
4. Additional Suggestions (if any)
                                                
**Make the advice SHORT, CLEAR AND ACTIONABLE. GIVE ALL THE ADVICE IN BULLET POINTS AND IN CATEGORISED MANNER. 
Give all Advice in a categorised manner and try to mention using CRS Score where applicable.**
USE STEP WISE REASONING TO ARRIVE AT THE FINAL ANSWER. 
USE STEPS TO EXPLAIN THE ACTIONABLE ADVICE IN A CHRONOLOGICAL MANNER.
                                                
CONTEXT:                                        
CRS SCORE LOOKUP TABLE:
# Comprehensive Ranking System Lookup Table

## Core/Human Capital Factors;
You will earn points as if you  **don’t**  have a spouse or partner if:
-   they are not coming with you to Canada, or.
-   they are a Canadian citizen or permanent resident.
 
|*_Factors_*|_*Points with Spouse*_  |_*Points without Spouse*_|
|--|--|--|
|Age|100|110|
|Level of Education|140|150|
|Official Language Proficiency|150|160|
|Canadian Work Experience|70|80|

## Detailed Points Breakdown

|*_Age_*| _*Points with Spouse*_  |_*Points without Spouse*_|
|--|--|--|
|17years or less|0|0
|18years|90|99
|19years|95|105
|20-29years|100|110
|30years|95|105
|31years|90|99
|32years|85|94
|33years|80|88
|34years|75|83
|35years|70|77
|36years|65|72
|37years|60|66
|38years|55|61
|39years|50|55
|40years|45|50
|41years|35|39
|42years|25|28
|43years|15|17
|44years|5|6
|45years or more|0|0|

## Level of Education Points Breakdown
|*_Level of Education_*| _*Points with Spouse*_  |_*Points without Spouse*_|
|--|--|--|
|Less than secondary school|0|0|
|Secondary diploma (high school graduation)|28|30|
|One-year degree, diploma or certificate from a university, college, trade or technical school, or other institute|84|90|
|Two-year program at a university, college, trade or technical school, or other institute|91|98|
|Bachelor's degree OR a three or more year program at a university, college, trade or technical school, or other institute|112|120|
|Two or more certificates, diplomas, or degrees. One must be for a program of three or more years|119|128|
|Master's degree, OR professional degree needed to practice in a licensed profession (For “professional degree,” the degree program must have been in: medicine, veterinary medicine, dentistry, optometry, law, chiropractic medicine, or pharmacy.)|126|135|
|Doctoral level university degree (Ph.D.)|140|150|

## Official Language Proficiency Points Breakdown
Maximum points for each ability (reading, writing, speaking and listening):
-   32 with a spouse or common-law partner
-   34 without a spouse or common-law partner

_Canadian Language Benchmark *CLB* Level_|_With a Spouse_|_Without a Spouse_|
|--|--|--|
|Less than CLB 4|0|0
|CLB 4 or 5|6|6|
|CLB 6|8|9|
|CLB 7|16|17|
|CLB 8|22|23|
|CLB 9|29|31|
|CLB 10 or more|32|34|

### Second Language Points Breakdown
_Canadian Language Benchmark *CLB* Level_|_With a Spouse_|_Without a Spouse_|
--|--|--|
|CLB 4 or less|0|0|
|CLB 5 or 6|1|1|
|CLB 7 or 8|3|3|
|CLB 9 or more|6|6|

## Canadian Work Experience (*Only Applicable if you have work experience in Canada*)
Canadian Work Experience| With a Spouse| Without a Spouse|
|--|--|--
|None or less than a year|0|0|
|1 year|35|40|
|2 years|46|53|
|3 years|56|64|
|4 years|63|72|
|5 years or more|70|80|

## Skill Transferability Factors;
|Education | Points
--|--
With good official languages proficiency and a post-secondary degree|50
With Canadian work experience and a post-secondary degree|50
___
Foreign Work Experience|Points
--|--
With good official language proficiency and foreign work experience| 50
With Canadian Work Experience and Foreign Work Experience | 50
---
Certificate of Qualification | Points |
--|--
With good/strong official languages proficiency and a certificate of qualification | 50

## Additional Points
|Factor| Points
--|--
|Brother or sister living in Canada (18 years or older, citizen or permanent resident)|15
|French language skills| 50 (ONLY if you have strong French skills and moderate English skills)
|Post-secondary education in Canada|30 (Only if you have completed a degree, diploma or certificate from a *CANADAIAN* post-secondary institution)
|Provincial or territorial nomination|600

Rethink all your answers based on the above CRS Score table wherever applicable. and answer conservatively.
                                       
                                                
Question:
{question}
""")

# Build RAG chain manually (modern pattern)
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | systemprompt
    | llm
)

# =============================
# API Endpoints
# =============================
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Visa Desk RAG API",
        "version": "1.0.0",
        "status": "active"
    }


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the visa information system",
    description="Send applicant data and questions about Canadian immigration"
)
async def query_documents(request: QueryRequest):
    """Process a query using the RAG system."""
    try:
        # Create a structured analysis request
        if request.data:
            # Format the profile data in a clear, structured way
            profile_summary = []
            if "name" in request.data:
                profile_summary.append(f"Applicant: {request.data['name']}")
            if "age" in request.data:
                profile_summary.append(f"Age: {request.data['age']}")
            if "education" in request.data:
                edu = request.data["education"]
                profile_summary.append(
                    f"Education: {edu.get('highest_degree')} in {edu.get('field_of_study')}, {edu.get('year_of_completion')}"
                )
            if "work_experience" in request.data:
                work = request.data["work_experience"]
                profile_summary.append(
                    f"Work: {work.get('current_role')}, {work.get('total_years')} years experience"
                )
            if "language_proficiency" in request.data:
                lang = request.data["language_proficiency"].get("IELTS", {})
                profile_summary.append(
                    f"IELTS: L:{lang.get('listening')} R:{lang.get('reading')} W:{lang.get('writing')} S:{lang.get('speaking')}"
                )
            if "proof_of_funds" in request.data:
                funds = request.data["proof_of_funds"]
                profile_summary.append(f"Available Funds: CAD {funds.get('converted_to_cad')}")
            
            context_question = "Please analyze the following profile for Canadian immigration:\n\n"
            context_question += "\n".join(profile_summary) + "\n\n"
            if request.question:
                context_question += f"Specific Question: {request.question}\n"
            context_question += "\nBased on the provided profile, what are the best immigration pathways available?"
        else:
            context_question = request.question or "What are the general requirements for Canadian immigration?"

        response = rag_chain.invoke(context_question)
        return QueryResponse(
            answer=response.content,
            sources=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

    # http://127.0.0.1:8000/docs

