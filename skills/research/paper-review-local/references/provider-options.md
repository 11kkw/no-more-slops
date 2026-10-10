# Verified integration options (2026-10-10)

## Free-first deployment

Current automatic checks run locally using pypdf (BSD-3-Clause), RapidFuzz (MIT) and ChkTeX (GPL). Compare only supplied open/full-text sources; no accounts, paid service or manuscript upload. Exact phrases and sentence string similarities flag candidates; they do not detect every paraphrase or scan a scholarly database.

- https://github.com/rapidfuzz/RapidFuzz
- https://www.nongnu.org/chktex/

Paperpal's official free plan offers 7,000 words/calendar month, an overall similarity score and top three sources. Full downloadable/side-by-side reports require upgrading. This is an optional manual web check, not an installed free API/MCP integration. No account or manuscript upload was made. Do not claim a public free API is available: that was not verified.

- https://paperpal.com/tools/plagiarism-checker
- https://support.paperpal.com/support/solutions/articles/3000124500-how-many-pages-or-words-can-i-check-every-month-for-plagiarism-

WCopyfind is free/open-source Windows software for matching supplied documents. It is a manual alternative; it does not provide a global academic database or a verified MCP. It is not installed on this Linux writing server.

- https://plagiarism.bloomfieldmedia.com/software/wcopyfind-instructions/

TeXtidote with bundled LanguageTool is a free local grammar/style option. It requires Java 9+; the current server lacks Java, so it is recommended but not installed or reported as passing. ChkTeX checks typography, not grammar or arXiv acceptance.

- https://github.com/sylvainhalle/textidote

## Paid connectors, disabled

Plugin directory searches for iThenticate/Turnitin/plagiarism and Copyleaks/Originality returned no relevant plagiarism plugin. This is not an exhaustive directory result.

Copyleaks has an official Streamable HTTP MCP endpoint at `https://extensions.copyleaks.com/mcp`, with OAuth 2.1 PKCE. Official documentation requires a paid app account (subscription, prepaid or organization balance); an API key alone does not authorize this MCP. Account is currently unavailable. Keep this connector disabled; do not invent tokens or claim it is connected.

- https://docs.copyleaks.com/mcp/overview
- https://docs.copyleaks.com/mcp/client-setup

iThenticate documents a supported XML-RPC API requiring an eligible administrator account. Turnitin Core API has separate licensed integration instances. Neither is connected. Report-viewer scraping is not an integration strategy.

- https://guides.ithenticate.com/hc/en-us/articles/27834631555725-iThenticate-API
- https://developers.turnitin.com/turnitin-core-api/faq

Originality.ai offers a plagiarism API, but its inspected MCP technical overview describes AI-text detection. Do not assume that MCP exposes plagiarism scanning without actual tool discovery.

- https://help.originality.ai/en/article/api-1a1ea3s/
- https://mcp.docs.originality.ai/

Similarity and plagiarism are different judgments. A threshold must not automatically certify or reject a manuscript. Local phrase coverage is not comparable with an external database's similarity score.

- https://www.crossref.org/documentation/similarity-check/similarity-report-understand/
- https://info.arxiv.org/help/submit/index.html
