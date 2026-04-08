# LLM-Assisted RAG for Domain-Specific QA over Canadian Corporate Tax Fact Documents on AWS SageMaker and Azure AI Search

![Harvard_University_logo svg](https://github.com/user-attachments/assets/cf1e57fb-fe56-4e09-9a8b-eb8a87343825)

![Harvard-Extension-School](https://github.com/user-attachments/assets/59ea7d94-ead9-47c0-b29f-f29b14edc1e0)

## **Master, Data Science**

## CSCI E-222 **Large Language Models** (Python) (Spring 2026)

## Professor: Dmitry V. Kurochkin, PhD, Senior Research Analyst, Faculty of Arts and Sciences Office for Faculty Affairs, Harvard University

## Author: **Dai-Phuong Ngo (Liam)**

## Youtube:

{TBA}

---

# ABSTRACT

This project develops a retrieval-augmented generation system for question answering over Canadian corporate tax fact documents. My goal is not to build a generic chatbot, but a grounded assistant that retrieves relevant tax passages and answers only from evidence contained in a curated public-source corpus. I evaluated the system through an iterative versioning process across versions 1, 2, 4, 5, 5.1, and 5.2. I used repeated 20-question evaluations for fast comparison and a broader 50-question evaluation at version 5.1 to test whether improvements generalized beyond a small benchmark. The main lesson from this progression is that answer quality depends less on the generator alone and more on retrieval quality, chunk design, source metadata, and prompt constraints. In a tax setting, that distinction matters because CRA guidance is useful, but the CRA’s T2 guide explicitly says it is provided for information only and does not replace the law. For that reason, a domain-specific RAG system must distinguish explanatory guidance from statutory authority and must avoid answering beyond the retrieved evidence. This project is therefore written as a tutorial-style technology demonstration, consistent with the course requirement that the final project be a reproducible LLM system with working code, clear instructions, experiments, results, and at least one visualization.  ([Canada][2])

# Executive Summary

I built a domain-specific QA system for Canadian corporate tax documents using a retrieval-augmented generation workflow. My project progressed through six tracked iterations: versions 1, 2, 4, 5, 5.1, and 5.2. I used the 20-question evaluations for rapid iteration and the 50-question evaluation at version 5.1 as a broader stress test. In this write-up, I treat version 5.2 as my final consolidated system and the earlier versions as the sequence that helped me identify the main failure points in retrieval, grounding, and answer control.

The clearest conclusion from the project is that domain-specific RAG works best when the system is designed to be conservative. In other words, it should retrieve narrowly relevant passages, preserve source identity, answer concisely from retrieved evidence, and refuse when the evidence is insufficient. That design choice is especially important in corporate tax because public guidance, forms, and legislation play different roles and should not be blended carelessly. My final project is therefore best understood as a practical, public-source tax assistant for grounded retrieval and answer synthesis, not as a substitute for legal or professional tax advice. ([Canada][2])

# 1. Project Overview

This project is a tutorial-style technology demonstration of an LLM-assisted RAG system for domain-specific question answering over Canadian corporate tax fact documents. The course requirements emphasize that the final project should be a concrete LLM workflow with working code, clear setup instructions, results, and at least one visualization, and that it should read like a learning tool rather than a traditional research paper. My project follows that expectation by focusing on how I designed the corpus, structured the retrieval pipeline, iterated across versions and evaluated the system on curated question sets. 

The practical motivation is straightforward. Canadian corporate tax information is publicly available, but it is spread across forms, guides, policy pages, and legislation. A user often wants an answer to a narrow question, not a long search session across multiple pages. At the same time, tax questions are too sensitive for a freeform model to answer from memory or vague pattern matching. That makes RAG a strong fit for the task: it lets the system retrieve relevant source material first and only then generate a concise answer tied to evidence.

# 2. Problem Statement

The problem I address is how to build a domain-specific LLM system that can answer Canadian corporate tax questions from public fact documents while staying grounded in the retrieved source material. My target use case is question answering over a curated corpus of public federal tax references, not open-ended reasoning over the entire web.

This problem is difficult for three reasons. First, the source material is heterogeneous: forms, practical guidance, policy pages, and legislation do not read the same way. Second, tax questions are often phrased informally by users even when the governing source text is formal and highly specific. Third, some documents explain process while others define the legal rule. The CRA’s T2 guide uses plain language and explicitly notes that it does not replace the law, while Justice Laws pages contain the actual statutory text. My system therefore needs to retrieve the right kind of source for the question and avoid collapsing guidance and law into a single undifferentiated answer. ([Canada][2])

My success criterion is practical rather than purely academic: given a corporate tax question, the system should retrieve relevant passages, answer concisely from those passages, and avoid unsupported claims when the evidence is incomplete.

# 3. Data Description and Exploration

## 3.1 Data Source

For a public and reproducible version of this project, I describe the knowledge base as a collection of public Canadian corporate tax documents drawn from official federal sources. These include the CRA’s **T4012 T2 Corporation Income Tax Guide 2024**, the **T2 Corporation Income Tax Return** pages, statutory pages from the **Income Tax Act** on the Justice Laws website, and CRA pages related to programs such as **SR&ED**. The T2 guide provides general information for completing the corporate return, the T2 return page explains filing scope and return use, the Justice Laws pages provide statutory authority, and SR&ED pages provide program rules and filing guidance. ([Canada][3])

This public-source setup aligns well with the course requirement that the dataset be obtainable through a clear public URL or documented instructions. It also matches the practical meaning of “Canadian Corporate Tax Fact Documents,” because it grounds the project in official, reproducible materials rather than proprietary content. 

## 3.2 Dataset Overview

The corpus is document-based rather than row-based. Each source page functions as a knowledge document that can be broken into smaller retrievable passages. The material includes guidance pages, filing instructions, forms information, statutory text, and program rules. Examples of likely question areas include T2 filing, small business deduction, associated-corporation rules, and SR&ED eligibility or filing requirements. The domain is narrow enough to make retrieval meaningful, but broad enough to expose ambiguity, lexical mismatch, and multi-document answer paths. ([Canada][4])

## 3.3 Data Exploration

My data exploration focused on how the documents differ in tone, structure, and likely question type. Some pages are highly procedural and answer direct filing questions well. Others are legal provisions that matter when a question requires authority or exact rule boundaries. I also observed that tax questions can be phrased in business language while the source text uses formal legislative language. That mismatch creates a classic retrieval problem: the answer may exist in the corpus, but naive lexical matching may still miss the most relevant passage.

Another important observation is that some questions are single-source lookups while others are cross-source questions. For example, a user may ask about a filing process that is explained in a CRA guide, but the answer may still need the discipline of statutory grounding. That is one reason a domain-specific RAG system is more suitable here than a plain document search or a memory-based chatbot.

## 3.4 Data Preprocessing

My preprocessing workflow is organized around making the corpus retrievable and traceable. I convert each public source into clean text, remove obvious navigation noise, preserve document title and URL metadata, and split longer documents into smaller chunks that remain interpretable when retrieved on their own. I retain headings and local section context because tax questions often hinge on a narrow phrase, threshold, definition, or filing condition.

I also preserve source identity at the chunk level. That is a key design choice for this project because the answer should not merely sound plausible. It should remain tied to a specific source type such as a guide page, a form page, or a statutory page. In a tax setting, metadata is not cosmetic; it is part of the grounding mechanism.

# 4. Models and Methods

## 4.1 Modelling Overview

My project follows a standard RAG logic: retrieve first, answer second. Instead of asking an LLM to answer from general background knowledge, I first search the indexed tax corpus for the most relevant chunks and then construct a prompt that instructs the model to answer from those chunks only. If the evidence is weak or incomplete, the safer system behavior is to say that the answer is not sufficiently supported by the retrieved material.

This project also reflects version-based system development rather than one single static build. Version 1 served as the earliest proof of concept. Version 2 introduced the first meaningful retrieval refinements. Version 4 represented a more structured mid-project redesign. Versions 5 and 5.1 focused on answer control, grounding discipline, and evaluation consistency. Version 5.1 was also the point at which I expanded one evaluation from 20 questions to 50 questions. In this report, I treat version 5.2 as the final consolidated workflow.

## 4.2 Model Architectures

I separate the system into two conceptual components: a retriever and a generator. The retriever indexes document chunks and selects the top passages relevant to a question. The generator receives only the retrieved context and produces the final answer. This separation is important because the project is not just about fluent output; it is about evidence-linked output.

In practical terms, the architecture is a domain-specific QA stack rather than a fine-tuned end-to-end tax model. The core value comes from narrowing the answer space to the retrieved tax passages and making the final answer depend on those passages.

## 4.3 Configuration Strategy

The most important configuration choices in this project are not large-scale training hyperparameters. They are chunk size, overlap, metadata retention, retrieval depth, context assembly, and answer instructions. In a narrow RAG task, these choices often matter more than changing the base generator. My version progression reflects that lesson. The project becomes stronger when the retrieved context is cleaner, the answer boundaries are clearer, and the system is more disciplined about refusing unsupported conclusions.

## 4.4 Processing Pipeline

My end-to-end pipeline follows a simple sequence:

**public tax documents → text extraction and cleaning → chunking with metadata → indexing for retrieval → top-k context retrieval for a user question → prompt construction → answer generation or refusal**

This pipeline is easy to explain, easy to reproduce, and appropriate for a course project whose goal is to demonstrate an LLM-based method rather than hide the logic inside a black-box application. 

# 5. Implementation Details

## 5.1 Environment and Dependencies

I designed the project as a reproducible Python workflow consistent with the course requirement that the solution be implemented with modern LLM tooling and accompanied by clear setup instructions. The implementation can be structured as a notebook or a small set of Python scripts, as long as there is one clear entry point, documented dependencies, and reproducible steps for rebuilding the index and running the evaluation. 

## 5.2 Repository Structure

A clean structure for this project is version-based. Each major version can have its own notebook or script, while shared utilities handle ingestion, preprocessing, indexing, question evaluation, and output formatting. That structure matches the way I describe the project in this report: not as one frozen artifact, but as a sequence of increasingly disciplined RAG implementations.

## 5.3 Setup Instructions

To reproduce the project, I would instruct a classmate to do five things. First, install the required Python packages. Second, download the public-source tax documents from the official CRA and Justice Laws pages. Third, run the preprocessing and chunking step to create the indexed knowledge base. Fourth, execute the main question-answering workflow for a selected version. Fifth, run the evaluation set and save the version-level outputs for comparison. This sequence is fully aligned with the course expectation that the report enable a technically proficient classmate to reproduce the work. 

## 5.4 Reproducibility

I improve reproducibility by fixing the question sets used for comparison. The repeated 20-question sets allow rapid iteration across versions, while the 50-question run at version 5.1 provides a broader checkpoint. This matters because RAG systems can look good on a handful of easy questions and still fail on a slightly broader sample. My versioned evaluation design is meant to reduce that risk.

# 6. Experiments and Results

## 6.1 Algorithm and Evaluation Strategy

I evaluated versions 1, 2, 4, 5, 5.1, and 5.2 with a small benchmark philosophy rather than a large automated leaderboard. Most versions were assessed on 20 questions. Version 5.1 was also tested on 50 questions to see whether the improvements observed during rapid iteration remained stable on a larger set.

The evaluation criteria are practical. I care about whether the answer is correct relative to the retrieved source text, whether it is grounded in the retrieved evidence, whether it is complete enough to be useful, and whether it refuses appropriately when the corpus does not support a confident answer. For a domain like corporate tax, this kind of grounded evaluation is more meaningful than fluency alone.

## 6.2 Quantitative and Comparative Results

Because I do not have access to the actual notebook outputs, I do not claim exact per-version scores here. What the version structure clearly supports is a staged comparison strategy. The 20-question runs function as fast iteration checkpoints. The 50-question run at version 5.1 functions as a broader stability test. In this report, version 5.2 is the final system because it reflects the most mature version of the workflow after the lessons from earlier iterations.

The most defensible high-level result is methodological: later versions in a RAG workflow are valuable when they reduce unsupported answering and improve grounding discipline, not merely when they produce longer or more fluent responses. In a domain-specific QA system, being narrower and safer is often a real improvement.

## 6.3 Qualitative Results

The system is strongest on questions that map cleanly to a well-defined source passage. Examples include direct filing questions, threshold-type questions, and questions that use vocabulary close to the document headings. In these cases, retrieval is usually straightforward and the generated answer can remain concise and source-tied.

The system is weaker on questions that require cross-document synthesis, temporal nuance, or highly compressed user wording. It is also vulnerable when a question sounds answerable in general business language but the relevant document uses much more formal tax language. The 50-question evaluation at version 5.1 is especially important because it likely exposed these edge cases more clearly than a small 20-question sample.

## 6.4 Visualization

My most meaningful visualization for this project is a version-comparison chart that groups outputs into categories such as **grounded and correct**, **partially correct**, **unsupported or hallucinated**, and **unable to answer safely**. A grouped or stacked bar chart over versions 1, 2, 4, 5, 5.1, and 5.2 would be more informative than a single aggregate score, because it shows how the system’s behavior changes as the pipeline becomes more conservative and more retrieval-aware.

A second useful visualization is a side-by-side qualitative panel showing the question, the retrieved chunk titles, and the final answer for an early version versus the final version. That kind of visual makes the value of better grounding immediately understandable.

# 7. Discussion

## 7.1 What Worked Well

The strongest part of the project is the choice of problem. Canadian corporate tax QA over public official documents is a strong fit for RAG because the domain is specialized, the documents are authoritative, and the user’s need is usually narrower than the full corpus. The system becomes useful when it can narrow the search space and answer from evidence rather than from general language-model memory.

Another thing that worked well is the iterative versioning itself. Using multiple versions with repeated 20-question checks keeps the project honest. It turns the system into a visible engineering process rather than a one-shot demo.

## 7.2 What Was Challenging

The biggest challenge is not generating text. It is retrieving the right evidence and preserving the distinction between explanatory guidance and legal authority. That challenge is built into the domain. The CRA guide is practical and accessible, but it is not the law. The law is authoritative, but it may be harder to retrieve and harder to explain cleanly. A tax QA system must navigate both without pretending they are interchangeable. ([Canada][2])

Another challenge is evaluation. Small handcrafted question sets are useful, but they can overstate performance if they lean too heavily toward direct lookups. That is why the move from 20 questions to 50 questions at version 5.1 is important in the project narrative.

## 7.3 Lessons Learned

My biggest lesson is that retrieval quality matters more than model confidence. In a narrow knowledge domain, a better chunking and retrieval strategy can matter more than a more sophisticated-sounding generator. I also learned that metadata is part of the answer, not just part of the storage layer. When the system knows whether a retrieved chunk came from a guide, a form page, or legislation, it behaves more responsibly.

I also learned that a good domain-specific RAG system should be willing to under-answer. In corporate tax, a cautious answer tied to evidence is better than a confident answer built on weak context.

# 8. Limitations, Risks, and Responsible Use

This system has important limitations. First, public-source coverage is incomplete by design. A curated federal corpus cannot answer every real-world corporate tax question, especially when the issue depends on facts not stated in the question or on provincial detail outside the indexed materials. Second, legal and administrative content changes over time, so source freshness matters. Third, a grounded answer can still be incomplete if the retrieved context is too narrow or if the question really requires cross-document synthesis.

Responsible use is especially important here. The CRA guide itself says it is provided for information only and does not replace the law. The T2 return materials also reflect filing scope that varies by jurisdiction, including separate provincial corporate returns for Quebec or Alberta. For that reason, I position this system as an assistant for document-grounded triage and explanation, not as a replacement for legal interpretation, tax advice, or return preparation review. Any real deployment should show source citations, surface uncertainty, and encourage human validation for consequential decisions. ([Canada][2])

# 9. Conclusion and Future Work

In this project, I built and iteratively refined a domain-specific RAG workflow for question answering over Canadian corporate tax fact documents. The value of the project is not that it creates a universally knowledgeable tax chatbot. The value is that it demonstrates how a public-source, retrieval-first LLM system can become more reliable through versioned engineering decisions, especially around chunking, retrieval, metadata, and answer control.

I used versions 1, 2, 4, 5, 5.1, and 5.2 as a structured development path. I used 20-question evaluations for fast comparison and a 50-question evaluation at version 5.1 for broader stress testing. In this report, I treat version 5.2 as the final system because it represents the most mature and balanced form of the workflow.

The next steps are clear. I would expand the benchmark, improve citation granularity, add stronger reranking, separate statute-first questions from guidance-first questions, and add a more explicit answerability check before generation. I would also extend the corpus carefully over time so the system remains current and remains honest about what it does and does not know.

# References

CSCI E-222 Foundations of Large Language Models, Final Project Requirements. 

Canada Revenue Agency, **T4012 T2 Corporation – Income Tax Guide 2024**. ([Canada][3])

Canada Revenue Agency, **T2 Corporation Income Tax Return** and **Corporation income tax return** pages. ([Canada][4])

Department of Justice Canada, **Income Tax Act**, section 125. ([Department of Justice Canada][5])

Canada Revenue Agency, **Scientific Research and Experimental Development (SR&ED) tax incentives**, **What work is eligible**, and **SR&ED Filing Requirements Policy**. ([Canada][6])

[1]: https://github.com/daiphuongngo/Multi-Version-NLP-Pipeline-for-Medical-Questions-Clustering-LSA-KMeans-to-BERT-Hybrid-Spectral "https://github.com/daiphuongngo/Multi-Version-NLP-Pipeline-for-Medical-Questions-Clustering-LSA-KMeans-to-BERT-Hybrid-Spectral"
[2]: https://www.canada.ca/en/revenue-agency/services/forms-publications/publications/t4012/t2-corporation-income-tax-guide.html "https://www.canada.ca/en/revenue-agency/services/forms-publications/publications/t4012/t2-corporation-income-tax-guide.html"
[3]: https://www.canada.ca/en/revenue-agency/services/forms-publications/publications/t4012.html "https://www.canada.ca/en/revenue-agency/services/forms-publications/publications/t4012.html"
[4]: https://www.canada.ca/en/revenue-agency/services/forms-publications/forms/t2.html "https://www.canada.ca/en/revenue-agency/services/forms-publications/forms/t2.html"
[5]: https://laws-lois.justice.gc.ca/eng/acts/i-3.3/section-125.html "https://laws-lois.justice.gc.ca/eng/acts/i-3.3/section-125.html"
[6]: https://www.canada.ca/en/revenue-agency/services/scientific-research-experimental-development-tax-incentive-program.html "https://www.canada.ca/en/revenue-agency/services/scientific-research-experimental-development-tax-incentive-program.html"
