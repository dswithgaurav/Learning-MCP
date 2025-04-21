from fastapi import FastAPI, Request, HTTPException, Depends
import uvicorn
import openai
import os
import sys
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union, List
import requests
from functools import lru_cache
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Settings:
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.google_api_key = os.environ.get("SEARCH_API", "")
        self.search_engine_id = os.environ.get("SEARCH_INDEX", "")

        logger.info(f"OPENAI API KEY set : {'Yes' if self.openai_api_key else 'No'}")
        logger.info(f"GOOGLE API KEY set : {'Yes' if self.google_api_key else 'No'}")
        logger.info(f"Search Engine set : {'Yes' if self.search_engine_id else 'No'}")


@lru_cache
def get_settings():
    return Settings()


class BaseReqest(BaseModel):
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    search_engine_id: Optional[str] = None


class ParseFileRequest(BaseReqest):
    file_path: str


class PlagiarismRequest(BaseReqest):
    text: str
    similarity_threshold: Optional[int] = 40


class GradeRequest(BaseReqest):
    text: str
    rubric: str
    model: Optional[str] = "gpt-3.5-turbo"


class FeedbackRequest(BaseReqest):
    text: str
    rubric: str
    model: Optional[str] = "gpt-3.5-turbo"


class ErrorResponse(BaseModel):
    detail: str


class GradeResponse(BaseModel):
    grade: str


class FeedbackResponse(BaseModel):
    feedback: str


class PlagiarismResult(BaseModel):
    url: str
    similarity: int


class PlagiarismResponse(BaseModel):
    results: List[PlagiarismResult]


# == FastAPI Setup ==
app = FastAPI(
    title="Assignment Grader API",
    description="API for Parsing, grading and checking plagiarism in academic assignements",
    version="1.0.0",
    responses={500: {"model": ErrorResponse}},
)


@app.get("/")
async def root():
    return {"message": "Assignment Grader API", "status": "Running", "version": "1.0.0"}


def get_api_keys(request, settings):
    openai_key = getattr(request, "openai_api_key", None) or settings.openai_api_key
    google_key = getattr(request, "google_api_key", None) or settings.google_api_key
    search_id = getattr(request, "search_engine_id", None) or settings.search_engine_id

    return {
        "openai_api_key": openai_key,
        "google_api_key": google_key,
        "search_engine_id": search_id,
    }


# == File Parsing ===
async def parse_pdf(file_path: str) -> str:
    try:
        import fitz

        doc = fitz.open(file_path)
        return "\n".json([page.get_text() for page in doc])
    except ImportError:
        raise HTTPException(status_code=500, detail="PyMuPDF not Installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing PDF : {str(e)}")


async def parse_docx(file_path: str) -> str:
    try:
        from docx import Document

        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except ImportError:
        raise HTTPException(status_code=500, detail="python-docx not Installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing DOCX : {str(e)}")


@app.post("/tools/parse_file", response_model=str)
async def parse_file(
    request: ParseFileRequest, settings: Settings = Depends(get_settings)
):
    print("hello World")
    try:
        file_path = request.file_path

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found {file_path}")
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == ".pdf":
            return await parse_pdf(file_path=file_path)
        elif ext == ".docx":
            return await parse_docx(file_path=file_path)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file format: {ext}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error parsing file : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing file:{str(e)}")


# == Plagiarism Check
@app.post("/tools/check_plagiarism", response_model=PlagiarismResponse)
async def check_plagiarism(
    request: PlagiarismRequest, settings: Settings = Depends(get_settings)
):
    try:
        keys = get_api_keys(request=request, settings=settings)

        if not keys["google_api_key"] or not keys["search_engine_id"]:
            raise HTTPException(
                status_code=500,
                detail="Google API Key or Search Engine Id not configured.",
            )

        from fuzzywuzzy import fuzz

        text = request.text
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text Cannot Be Empty")

        query = text[:300].replace("\n", " ").strip()
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": keys["google_api_key"],
            "cx": keys["search_engine_id"],
            "q": query,
        }

        response = request.get(url, params=params, timeout=10)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Google API Error : {response.text}",
            )

        data = response.json()
        results = data.get("items", [])

        plagiarism_results = [
            PlagiarismResult(
                url=item["link"],
                similarity=fuzz.token_set_ratio(text, item.get("snippet", "")),
            )
            for item in requests
        ]

        plagiarism_results.sort(key=lambda x: x.similarity, reverse=True)

        threshold = requests.similarity_threshold or 0
        if threshold > 0:
            plagiarism_results = [
                r for r in plagiarism_results if r.similarity >= threshold
            ]
        return PlagiarismResponse(results=plagiarism_results)
    except ImportError:
        raise HTTPException(status_code=500, detail="fuzzywuzzy not installed.")
    except Exception as e:
        logger.error(f"Error checking plagiarism : {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error checking plagiarism:{str(e)}"
        )


# == Grading Function
async def call_openai_api(
    prompt: str, api_key: str, model: str = "gpt-3.5-turbo"
) -> str:
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API Key is not configured.")

    try:
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error : {str(e)}")


@app.post("/tools/grade_text", response_model=GradeResponse)
async def grade_text(request: GradeRequest, settings: Settings = Depends(get_settings)):
    try:
        text = request.text
        rubric = request.rubric
        model = request.model or "gpt-3.5-turbo"

        keys = get_api_keys(request=request, settings=settings)

        if not text.strip() or not rubric.strip():
            raise HTTPException(
                status_code=400, detail="Text or Rubric cannot be empty."
            )

        if not keys["openai_api_key"]:
            raise HTTPException(
                status_code=500, detail="OpenAI API key not configured."
            )

        prompt = f"""
                You are a academic grader. Grade the following assignment based on the rubric. Respond with only grade.
                Rubric : {rubric}
                Assignment : {text}
                """
        grade = await call_openai_api(
            prompt=prompt, api_key=keys["openai_api_key"], model=model
        )
        return GradeResponse(grade=grade)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error grading text : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error grading text: {str(e)}")


@app.post("/tools/generate_feedback", response_model=str)
async def get_feedback(
    request: FeedbackRequest, settings: Settings = Depends(get_settings)
):
    try:
        text = request.text
        rubric = request.rubric
        model = request.model or "gpt-3.5-turbo"

        keys = get_api_keys(request=request, settings=settings)

        if not text.strip() or not rubric.strip():
            raise HTTPException(
                status_code=400, detail="Text or Rubric cannot be empty."
            )

        if not keys["openai_api_key"]:
            raise HTTPException(
                status_code=500, detail="OpenAI API key not configured."
            )

        prompt = f"""
                You are a academic grader. Give constructive feedback to the student based on this rubric and assignement.
                Rubric : {rubric}
                Assignment : {text}
                """
        feedback = await call_openai_api(
            prompt=prompt, api_key=keys["openai_api_key"], model=model
        )
        return FeedbackResponse(feedback=feedback)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating feedback : {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating feedback: {str(e)}"
        )


# == Support for alternate URL Format
@app.post("/tool/{tool_name}")
async def tool_endpoint_singular(
    tool_name: str, request: Request, settings: Settings = Depends(get_settings)
):
    try:
        body = await request.json()

        if tool_name == "parse_file":
            req = ParseFileRequest(**body)
            return await parse_file(req, settings)
        elif tool_name == "check_plagiarism":
            req = PlagiarismRequest(**body)
            return await check_plagiarism(req, settings)
        elif tool_name == "grade_text":
            req = GradeRequest(**body)
            return await grade_text(req, settings)
        elif tool_name == "generate_feedback":
            req = FeedbackRequest(**body)
            return await get_feedback(req, settings)

    except HTTPException:
        return
    except Exception as e:
        logger.error(f"Error generating feedback : {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating feedback: {str(e)}"
        )


@app.post("/api/tools/{tool_name}")
async def tool_endpoint_api(
    tool_name: str, request: Request, settings: Settings = Depends(get_settings)
):
    return await tool_endpoint_singular(
        tool_name=tool_name, request=request, settings=settings
    )


if __name__ == "__main__":
    logger.info("Assignment Grader API")
    logger.info("Available Tools")
    logger.info("- /tools/parse_file")
    logger.info("- /tools/check_plagiarism")
    logger.info("- /tools/grade_text")
    logger.info("- /tools/generate_feedback")
    uvicorn.run(app, host="0.0.0.0", port=8088)
