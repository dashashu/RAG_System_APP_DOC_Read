"""
Fetch Confluence pages (Atlassian Cloud REST API) for RAG indexing.

Auth: CONFLUENCE_EMAIL + CONFLUENCE_API_TOKEN (https://id.atlassian.com/manage-profile/security/api-tokens).

Set CONFLUENCE_URL to the wiki root, e.g. https://your-site.atlassian.net/wiki
and CONFLUENCE_SPACE_KEYS to a comma-separated list of space keys to crawl.
"""

from __future__ import annotations

import html as html_module
import logging
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")


def _storage_to_plain(html_like: str) -> str:
    """Best-effort plain text from Confluence storage (XHTML-like) format."""
    if not html_like or not html_like.strip():
        return ""
    # Drop script/style blocks
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html_like)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = _TAG_RE.sub(" ", text)
    text = html_module.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class ConfluencePageChunk:
    """One logical page worth of text (will be split again by RAG chunker)."""

    title: str
    space_key: str
    page_id: str
    url: str
    body: str


def _api_base(wiki_root: str) -> str:
    root = wiki_root.rstrip("/")
    return f"{root}/rest/api"


def fetch_pages_for_spaces(
    wiki_root: str,
    email: str,
    api_token: str,
    space_keys: list[str],
    *,
    max_pages: int,
    batch_limit: int,
    request_timeout: float = 60.0,
) -> list[ConfluencePageChunk]:
    """
    List pages per space (type=page), expand body.storage, return plain-text bodies.
    """
    if not space_keys:
        return []

    base = _api_base(wiki_root)
    auth = (email, api_token)
    headers = {"Accept": "application/json"}
    out: list[ConfluencePageChunk] = []

    with httpx.Client(timeout=request_timeout) as client:
        for space_key in space_keys:
            if len(out) >= max_pages:
                break
            start = 0
            while len(out) < max_pages:
                params: dict[str, Any] = {
                    "spaceKey": space_key,
                    "type": "page",
                    "limit": min(batch_limit, max_pages - len(out)),
                    "start": start,
                    "expand": "body.storage,version",
                }
                url = f"{base}/content"
                try:
                    r = client.get(url, params=params, auth=auth, headers=headers)
                except httpx.RequestError as e:
                    logger.error("Confluence request failed for space %s: %s", space_key, e)
                    break
                if r.status_code == 401:
                    logger.error(
                        "Confluence HTTP 401: check CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN "
                        "and that the user can read space %s",
                        space_key,
                    )
                    break
                if r.status_code == 403:
                    logger.error("Confluence HTTP 403: forbidden for space %s", space_key)
                    break
                if r.status_code != 200:
                    logger.error(
                        "Confluence HTTP %s for space %s: %s",
                        r.status_code,
                        space_key,
                        (r.text or "")[:500],
                    )
                    break
                data = r.json()
                results = data.get("results") or []
                if not results:
                    break
                wiki_root_norm = wiki_root.rstrip("/")
                for item in results:
                    if len(out) >= max_pages:
                        break
                    page_id = str(item.get("id", ""))
                    title = (item.get("title") or "Untitled").strip()
                    body_obj = (item.get("body") or {}).get("storage") or {}
                    raw = body_obj.get("value") or ""
                    plain = _storage_to_plain(raw)
                    if not plain:
                        logger.debug("Skipping empty page %s (%s)", page_id, title)
                        continue
                    webui = (item.get("_links") or {}).get("webui") or ""
                    if webui.startswith("/"):
                        page_url = f"{wiki_root_norm}{webui}"
                    else:
                        page_url = urljoin(wiki_root_norm + "/", webui)
                    out.append(
                        ConfluencePageChunk(
                            title=title,
                            space_key=space_key,
                            page_id=page_id,
                            url=page_url,
                            body=plain,
                        )
                    )
                if len(results) < params["limit"]:
                    break
                start += len(results)
                time.sleep(0.15)  # light pacing between API pages

    return out


def parse_space_keys(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]
