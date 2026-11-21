---
title: Document Q&A
emoji: 📄
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Ask questions about PDFs, with cited pages
---

Upload English or French PDFs and ask a question. The answer cites the pages it comes from.

How it works: each page is searched as an image by a visual retriever (tomoro-colqwen3-embed-4b), the best 20 pages are
reranked by reading their text (Qwen3-Reranker-4B), and Qwen3-VL-8B answers from the top 5 pages, image and text. All
models are open (Apache 2.0). Pages without a text layer are read from their image only.
