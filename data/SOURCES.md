# Document sources

The LangChain documentation pages in this folder are not original work. They are included so
the chatbot has a realistic corpus to search, one large enough that retrieval has to actually
choose between chunks rather than return everything.

All were fetched on 2026-08-04 from <https://docs.langchain.com> and are reproduced verbatim.

| File | Source |
|---|---|
| `langchain-short-term-memory.md` | <https://docs.langchain.com/oss/python/langchain/short-term-memory> |
| `langchain-add-memory.md` | <https://docs.langchain.com/oss/python/langgraph/add-memory> |
| `langchain-persistence.md` | <https://docs.langchain.com/oss/python/langgraph/persistence> |
| `langchain-agentic-rag.md` | <https://docs.langchain.com/oss/python/langgraph/agentic-rag> |
| `langchain-v1-migration.md` | <https://docs.langchain.com/oss/python/migrate/langchain-v1> |

## License

LangChain's documentation is published in the [langchain-ai/docs](https://github.com/langchain-ai/docs)
repository under the MIT License. That license permits redistribution as long as its copyright
notice and permission notice travel with the copy, so the upstream license is reproduced in
full below.

The rest of this project is separately MIT licensed. See [../LICENSE](../LICENSE).

### Upstream license (langchain-ai/docs)

```text
MIT License

Copyright (c) 2025 LangChain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## A note on staleness

These are point-in-time snapshots. LangChain moves quickly, and the live documentation has
almost certainly changed since the fetch date above. Answers from this chatbot reflect the
docs as they were on 2026-08-04, not as they are today. Check the source URLs before relying
on anything it tells you.
