import logging
import os
import sys
import time
import warnings
import random
from pathlib import Path
from typing import Generic, List, Optional, TypeVar, Union

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl
from requests.exceptions import RequestException, ConnectionError, ConnectTimeout, ReadTimeout, Timeout

logger = logging.getLogger(__name__)

# Handle both module import and direct execution
try:
    from ..cache import cached
    from .data_model import (
        PaperSnippet,
        SemanticScholarAuthorData,
        SemanticScholarPaperData,
    )
except (ImportError, ValueError):
    # When running directly, add mcp_backend directory to path and import directly
    current_dir = Path(__file__).resolve().parent
    mcp_backend_dir = current_dir.parent
    if str(mcp_backend_dir) not in sys.path:
        sys.path.insert(0, str(mcp_backend_dir))
    # Import directly from the mcp_backend directory
    import cache
    cached = cache.cached
    from apis.data_model import (
        PaperSnippet,
        SemanticScholarAuthorData,
        SemanticScholarPaperData,
    )

# Try to load .env from multiple possible locations
# apis/ -> mcp_server/ -> sciconharness/ -> SciConBench/ (project root)
env_paths = [
    ".env",  # Current directory (works when CWD is project root)
    Path(__file__).resolve().parent.parent.parent.parent / ".env",  # SciConBench/.env
]
for env_path in env_paths:
    if Path(env_path).exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()

S2_API_KEY = os.getenv("S2_API_KEY")
TIMEOUT = int(os.getenv("API_TIMEOUT", 10))
S2_SEARCH_LIMIT = 100

S2_GRAPH_API_URL = "https://api.semanticscholar.org/graph/v1"
S2_RECOMMENDATIONS_API_URL = "https://api.semanticscholar.org/recommendations/v1"

# authors.authorId,authors.paperCount,authors.citationCount
S2_PAPER_SEARCH_FIELDS = "paperId,corpusId,url,title,abstract,authors,authors.name,year,venue,citationCount,openAccessPdf,externalIds,isOpenAccess"
S2_PAPER_CITATION_FIELDS = (
    "paperId,corpusId,contexts,intents,isInfluential,title,abstract,venue,year,authors"
)
S2_PAPER_REFERENCE_FIELDS = (
    "paperId,corpusId,contexts,intents,isInfluential,title,abstract,venue,year,authors"
)
S2_PAPER_RECOMMENDATION_FIELDS = "paperId,corpusId,title,abstract,year,venue"

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    total: int
    offset: int = Field(..., description="Offset for pagination")
    next: int = Field(..., description="Next page offset")
    data: List[T] = Field(..., description="Data items")


# This is because the Semantic Scholar API right now doesn't support
# pagination for snippets, so we will ignore the offset and limit.
class PaperSnippetApiResponse(BaseModel, Generic[T]):
    data: List[T] = Field(..., description="Data items")


# This is a subset of the supported query parameters for the Semantic Scholar API.
# We include the most important ones and remove some for clarity. (e.g., there are year and
# publicationDateOrYear, which can be confusing to the model).
class SemanticScholarSearchQueryParams(BaseModel):
    query: str = Field(
        ...,
        description="A plain-text search query string.",
        examples=["BERT"],
    )
    year: Optional[str] = Field(
        None,
        description="Restrict results to the given range of publication year (inclusive).",
        examples=["2015-2020", "2015-", "-2015"],
    )
    minCitationCount: Optional[int] = Field(
        None,
        description="Restrict results to only include papers with the minimum number of citations, inclusive.",
        examples=[100, 1000],
    )
    sort: Optional[str] = Field(
        None,
        description="Sort results by publicationDate and citationCount in ascending or descending order.",
        examples=[
            "citationCount:asc",
            "publicationDate:desc",
        ],
    )
    venue: Optional[str] = Field(
        None,
        description="Restrict results by venue. Input could be an ISO4 abbreviation.",
        examples=["ACL", "EMNLP"],
    )


################## NOT EXPLICITLY USED ##################
# fields: Optional[str] = Field(None, description="A comma-separated list of the fields to be returned.")
# publicationTypes: Optional[str] = Field(None, description="Restrict results by publication types, use a comma-separated list.")
# openAccessPdf: Optional[bool] = Field(None, description="Restrict results to only include papers with a public PDF.")
# publicationDateOrYear: Optional[str] = Field(None, description="""
# Restrict results to the given range of publication dates or years (inclusive).
# Accepts the format <startDate>:<endDate>. Prefixes supported for specific ranges.""")
# fieldsOfStudy: Optional[str] = Field(None, desc="Restrict results to given field-of-study, using the s2FieldsOfStudy paper field.")
# offset: Optional[int] = Field(0, description="Start with the element at this position in the list.")
# limit: Optional[int] = Field(100, description="The maximum number of results to return, must be <= 100.")


@cached()
def search_semantic_scholar_keywords(
    query_params: SemanticScholarSearchQueryParams,
    *,
    offset: int = 0,
    limit: int = 25,
    fields: str = S2_PAPER_SEARCH_FIELDS,
    timeout: int = TIMEOUT,
) -> ApiResponse[SemanticScholarPaperData]:

    res = requests.get(
        f"{S2_GRAPH_API_URL}/paper/search",
        params={
            "offset": offset,
            "limit": limit,
            "fields": fields,
            **query_params.model_dump(exclude_none=True),
        },
        headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
        timeout=timeout,
    )

    res.raise_for_status()

    results = res.json()

    # For each paper, if we know their external arxiv ids, we can construct the open access pdf link
    # if it is not already provided.
    if "data" in results:
        for paper in results["data"]:
            if paper.get("openAccessPdf") is None:
                if paper["externalIds"]:
                    if "ArXiv" in paper["externalIds"]:
                        paper["openAccessPdf"] = dict(
                            url=f"https://arxiv.org/pdf/{paper['externalIds']['ArXiv']}",
                        )
                    if "ACL" in paper["externalIds"]:
                        paper["openAccessPdf"] = dict(
                            url=f"https://www.aclweb.org/anthology/{paper['externalIds']['ACL']}.pdf",
                        )
    else:
        results["data"] = []

    return results


class SemanticScholarSnippetSearchQueryParams(BaseModel):
    query: str = Field(
        ...,
        description="A plain-text search query string.",
        examples=["BERT"],
    )
    year: Optional[str] = Field(
        None,
        description="Optional: Restrict results to the given range of publication year (inclusive). If not provided, searches all years without restriction.",
        examples=["2015-2020", "2015-", "-2015"],
    )
    paperIds: Optional[Union[str, List[str]]] = Field(
        None,
        description="Restricts results to snippets from specific papers. To specify papers, provide a comma-separated list of their IDs. You can provide up to approximately 100 IDs.",
        examples=[
            "649def34f8be52c8b66281af98ae884c09aef38b",
            "649def34f8be52c8b66281af98ae884c09aef38b,CorpusId:215416146",
        ],
    )
    venue: Optional[str] = Field(
        None,
        description="Restrict results by venue. Input could be an ISO4 abbreviation.",
        examples=["ACL", "EMNLP"],
    )


@cached()
def search_semantic_scholar_snippets(
    query_params: SemanticScholarSnippetSearchQueryParams,
    *,
    offset: int = 0,
    limit: int = 10,
    timeout: int = TIMEOUT,
    max_retries: int = 5,
    initial_backoff: float = 1.0,
) -> PaperSnippetApiResponse[PaperSnippet]:
    if not S2_API_KEY:
        raise ValueError(
            "S2_API_KEY environment variable is not set. "
            "Get a free key at https://api.semanticscholar.org/ and add it to your .env file."
        )

    if offset:
        warnings.warn(
            "Right now the API does not support pagination, so the offset will be ignored."
        )

    params = query_params.model_dump(exclude_none=True)
    if (query_params.paperIds or "paperIds" in query_params) and isinstance(
        query_params.paperIds, list
    ):
        params["paperIds"] = ",".join(query_params.paperIds)

    # Retry with exponential backoff for 429 errors
    for attempt in range(max_retries):
        try:
            res = requests.get(
                f"{S2_GRAPH_API_URL}/snippet/search",
                params={
                    # "offset": offset,
                    "limit": limit,
                    **params,
                },
                headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
                timeout=timeout,
            )
            
            # Parse response
            results = res.json()
            
            # Check for 429 errors (both in HTTP status and JSON response)
            is_rate_limit = False
            if res.status_code == 429:
                is_rate_limit = True
            elif results.get("code") == "429" or "Too Many Requests" in str(results.get("message", "")):
                is_rate_limit = True
            
            if is_rate_limit:
                if attempt < max_retries - 1:
                    wait_time = initial_backoff * (2 ** attempt) + random.uniform(10, 15) # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                    logger.warning("Semantic Scholar API rate limit exceeded (429). Waiting %.1f seconds before retry %d/%d...", 
                                wait_time, attempt + 1, max_retries)
                    time.sleep(wait_time)
                    continue
                else:
                    # Last attempt failed, log and return the error response
                    logger.error("Semantic Scholar API rate limit error after %d retries. Returning error response.", max_retries)
                    return results
            
            # Check for other HTTP errors
            if res.status_code != 200:
                # Retry on server errors (500+) and gateway errors (502, 503, 504)
                retryable_status_codes = {502, 503, 504}
                is_server_error = res.status_code >= 500
                is_retryable = is_server_error or res.status_code in retryable_status_codes
                
                if is_retryable and attempt < max_retries - 1:
                    wait_time = initial_backoff * (2 ** attempt) + random.uniform(10, 15)
                    logger.warning("Semantic Scholar API error %d. Waiting %.1f seconds before retry %d/%d...", 
                                res.status_code, wait_time, attempt + 1, max_retries)
                    time.sleep(wait_time)
                    continue
                # For non-retryable errors, raise
                res.raise_for_status()
            
            # Success - verify we have a proper response with data field
            if "data" in results:
                # Valid response with data
                return results
            elif "code" in results and results.get("code") != "429":
                # Error response that's not a 429, return it
                return results
            else:
                # Response without data field but no error code - might be valid empty response
                return results
                
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = initial_backoff * (2 ** attempt)
                    logger.warning("Semantic Scholar API HTTP 429 error. Waiting %.1f seconds before retry %d/%d...", 
                                wait_time, attempt + 1, max_retries)
                    time.sleep(wait_time)
                    continue
            # Check if it's a retryable server error
            if e.response:
                retryable_status_codes = {502, 503, 504}
                is_server_error = e.response.status_code >= 500
                is_retryable = is_server_error or e.response.status_code in retryable_status_codes
                if is_retryable and attempt < max_retries - 1:
                    wait_time = initial_backoff * (2 ** attempt) + random.uniform(10, 15)
                    logger.warning("Semantic Scholar API HTTP error %d. Waiting %.1f seconds before retry %d/%d...", 
                                e.response.status_code, wait_time, attempt + 1, max_retries)
                    time.sleep(wait_time)
                    continue
            # Re-raise if not retryable or if we've exhausted retries
            raise
        except (ReadTimeout, Timeout, ConnectTimeout) as e:
            # Retry on timeout errors
            if attempt < max_retries - 1:
                wait_time = initial_backoff * (2 ** attempt)
                logger.warning("Semantic Scholar API timeout error. Waiting %.1f seconds before retry %d/%d...", 
                            wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue
            raise
        except ConnectionError as e:
            # Retry on connection errors
            if attempt < max_retries - 1:
                wait_time = initial_backoff * (2 ** attempt)
                logger.warning("Semantic Scholar API connection error. Waiting %.1f seconds before retry %d/%d...", 
                            wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue
            raise
        except RequestException as e:
            # For other request exceptions, retry as well
            if attempt < max_retries - 1:
                wait_time = initial_backoff * (2 ** attempt) + random.uniform(10, 15)
                logger.warning("Semantic Scholar API request error. Waiting %.1f seconds before retry %d/%d...", 
                            wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue
            raise
    
    # If we've exhausted all retries, raise an exception
    raise Exception(f"Semantic Scholar API request failed after {max_retries} retries due to rate limiting")


def search_semantic_scholar_bulk_api(
    query_params: SemanticScholarSearchQueryParams,
    *,
    fields: str = S2_PAPER_SEARCH_FIELDS,
    timeout: int = TIMEOUT,
) -> ApiResponse[SemanticScholarPaperData]:

    res = requests.get(
        f"{S2_GRAPH_API_URL}/paper/search/bulk",
        params={
            "fields": fields,
            **query_params.model_dump(exclude_none=True),
        },
        headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
        timeout=timeout,
    )

    res.raise_for_status()

    results = res.json()

    # For each paper, if we know their external arxiv ids, we can construct the open access pdf link
    # if it is not already provided.
    if "data" in results:
        for paper in results["data"]:
            if paper.get("openAccessPdf") is None:
                if paper["externalIds"]:
                    if "ArXiv" in paper["externalIds"]:
                        paper["openAccessPdf"] = dict(
                            url=f"https://arxiv.org/pdf/{paper['externalIds']['ArXiv']}",
                        )
                    if "ACL" in paper["externalIds"]:
                        paper["openAccessPdf"] = dict(
                            url=f"https://www.aclweb.org/anthology/{paper['externalIds']['ACL']}.pdf",
                        )
    else:
        results["data"] = []

    return results


def download_paper_details(
    paper_id: str,
    *,
    fields: str = S2_PAPER_SEARCH_FIELDS,
    timeout: int = TIMEOUT,
):
    res = requests.get(
        f"{S2_GRAPH_API_URL}/paper/{paper_id}",
        params={
            "fields": fields,
        },
        headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
        timeout=timeout,
    )

    res.raise_for_status()
    results = res.json()
    return results


def download_paper_references(
    paper_id: str,
    *,
    offset: int = 0,
    limit: int = 100,
    fields: str = S2_PAPER_REFERENCE_FIELDS,
    timeout: int = TIMEOUT,
):
    res = requests.get(
        f"{S2_GRAPH_API_URL}/paper/{paper_id}/references",
        params={
            "offset": offset,
            "limit": limit,
            "fields": fields,
        },
        headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
        timeout=timeout,
    )
    results = res.json()
    return results


def download_paper_citations(
    paper_id: str,
    *,
    offset: int = 0,
    limit: int = 100,
    fields: str = S2_PAPER_CITATION_FIELDS,
    timeout: int = TIMEOUT,
):
    res = requests.get(
        f"{S2_GRAPH_API_URL}/paper/{paper_id}/citations",
        params={
            "offset": offset,
            "limit": limit,
            "fields": fields,
        },
        headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
        timeout=timeout,
    )
    results = res.json()
    return results


def download_paper_details_batch(
    paper_ids: List[str],
    *,
    fields: str = S2_PAPER_SEARCH_FIELDS,
    timeout: int = TIMEOUT,
):
    res = requests.post(
        f"{S2_GRAPH_API_URL}/paper/batch",
        params={"fields": fields},
        json={"ids": paper_ids},
        headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
        timeout=timeout,
    )
    results = res.json()
    return results


# Example usage:
if __name__ == "__main__":
    result1 = search_semantic_scholar_snippets(
        SemanticScholarSnippetSearchQueryParams(
            query="Fetch on measuring implicit norms and values in online communities",
        ),
    )
