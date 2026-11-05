"""
IBM Interview Trainer Agent — Extended Knowledge Base & RAG Pipeline
400+ curated interview Q&A across 10 roles, adaptive retrieval,
and resume ingestion support.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Extended Curated Knowledge Base (400+ entries) ───────────────────────────
SEED_DOCUMENTS: list[dict[str, Any]] = [
    # ════════════════════════════════════════════════════
    # SOFTWARE ENGINEERING
    # ════════════════════════════════════════════════════
    {"id":"se_001","role":"software_engineer","category":"technical","level":"all","content":"Q: Explain the difference between a process and a thread.\nA: A process is an independent execution unit with its own memory space; a thread is a lightweight unit within a process sharing its memory. Threads communicate faster but need synchronization to avoid race conditions.\nTip: Mention use cases — multi-threaded servers, async I/O, CPU-bound vs I/O-bound workloads."},
    {"id":"se_002","role":"software_engineer","category":"technical","level":"mid_senior","content":"Q: What is the CAP theorem?\nA: A distributed system can guarantee only two of: Consistency, Availability, Partition Tolerance. Systems like Cassandra favor AP; HBase favors CP.\nTip: Relate to a database choice you made in a real project."},
    {"id":"se_003","role":"software_engineer","category":"technical","level":"all","content":"Q: Explain SOLID principles.\nA: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion. They improve maintainability, testability, and reduce coupling in OOP designs.\nTip: Give a concrete code example for at least one principle."},
    {"id":"se_004","role":"software_engineer","category":"technical","level":"senior","content":"Q: How do you design a URL shortener like bit.ly?\nA: Use a hash function (MD5/Base62 encoding) on the URL, store in a key-value store (Redis + DB). Handle collisions with retries. Add a CDN for redirect speed, analytics with async logging.\nTip: Discuss scale: 100M URLs, 1B redirects/day."},
    {"id":"se_005","role":"software_engineer","category":"technical","level":"all","content":"Q: What is the difference between SQL and NoSQL databases?\nA: SQL databases are relational with ACID guarantees, good for structured data with complex queries. NoSQL offers flexible schemas, horizontal scaling, eventual consistency — suited for large-scale, unstructured data.\nTip: Mention when you'd choose each based on use case."},
    {"id":"se_006","role":"software_engineer","category":"technical","level":"mid_senior","content":"Q: Explain REST vs GraphQL.\nA: REST uses multiple endpoints with fixed data shapes; GraphQL uses a single endpoint where clients specify exactly what data they need, reducing over/under-fetching.\nTip: Discuss trade-offs: REST is simpler and cacheable; GraphQL is flexible for complex UIs."},
    {"id":"se_007","role":"software_engineer","category":"behavioral","level":"all","content":"Q: Describe a time you had to refactor a large codebase.\nA (STAR): Led migration of a monolith to microservices using the strangler-fig pattern. Wrote tests first, extracted services incrementally. Result: 40% faster deployments, 60% fewer incidents.\nTip: Quantify impact — numbers make answers memorable."},
    {"id":"se_008","role":"software_engineer","category":"technical","level":"all","content":"Q: What is Big O notation? Give examples.\nA: Big O describes algorithm time/space complexity as input grows. O(1) = hash lookup; O(log n) = binary search; O(n) = linear scan; O(n²) = nested loops.\nTip: Always mention the worst-case and explain why it matters for the interviewer's domain."},
    {"id":"se_009","role":"software_engineer","category":"technical","level":"mid_senior","content":"Q: How does garbage collection work in Java/Python?\nA: Python uses reference counting + cyclic GC. Java uses generational GC (Young/Old/Permanent gen) with algorithms like G1, ZGC. GC pauses are the key trade-off.\nTip: Mention when GC tuning matters — low-latency trading, real-time systems."},
    {"id":"se_010","role":"software_engineer","category":"situational","level":"all","content":"Q: Your production deployment broke the site. What do you do?\nA: 1) Immediately rollback or feature-flag off the change. 2) Notify stakeholders. 3) Reproduce in staging. 4) Fix root cause. 5) Post-mortem with blameless culture.\nTip: Show ownership and structured incident response."},

    # ════════════════════════════════════════════════════
    # DATA SCIENTIST
    # ════════════════════════════════════════════════════
    {"id":"ds_001","role":"data_scientist","category":"technical","level":"all","content":"Q: What is the bias-variance tradeoff?\nA: Bias = underfitting from wrong assumptions; Variance = overfitting from sensitivity to training data. Regularization, ensembles, and cross-validation help find the sweet spot.\nTip: Draw the U-shaped total error curve on a whiteboard."},
    {"id":"ds_002","role":"data_scientist","category":"technical","level":"senior","content":"Q: How do you handle class imbalance?\nA: SMOTE oversampling, undersampling, class_weight parameter, threshold tuning, use PR-AUC over ROC-AUC, collect more minority data.\nTip: Always ask the interviewer about the business cost of false positives vs negatives first."},
    {"id":"ds_003","role":"data_scientist","category":"technical","level":"all","content":"Q: Explain the difference between bagging and boosting.\nA: Bagging trains models in parallel on bootstrap samples (Random Forest); Boosting trains sequentially, each model correcting the previous (XGBoost, AdaBoost). Boosting reduces bias more; Bagging reduces variance.\nTip: Mention when each is preferred — boosting for tabular competitions, bagging for stability."},
    {"id":"ds_004","role":"data_scientist","category":"technical","level":"mid_senior","content":"Q: How do you detect and handle data leakage?\nA: Data leakage occurs when test-set information influences training. Prevention: always fit scalers/encoders on training data only; use proper time-series splits; check feature correlations with the target.\nTip: Give a real example — target encoding without holdout is a classic leakage."},
    {"id":"ds_005","role":"data_scientist","category":"technical","level":"all","content":"Q: What evaluation metrics would you use for a fraud detection model?\nA: Precision, Recall, F1 (since classes are imbalanced), PR-AUC, and business cost matrix. Accuracy alone is misleading (99% accuracy if 1% fraud).\nTip: Always tie metrics to business impact — cost of missed fraud vs. false alerts."},
    {"id":"ds_006","role":"data_scientist","category":"situational","level":"mid_senior","content":"Q: A key metric drops 20% overnight. What do you investigate?\nA: 1) Check data pipeline integrity (tracking bug?). 2) Segment by platform/region/cohort. 3) Review recent deployments/A-B tests. 4) Check external factors. 5) Formulate hypothesis, run analysis.\nTip: Show structured thinking — interviewers value process over quick answers."},
    {"id":"ds_007","role":"data_scientist","category":"technical","level":"senior","content":"Q: Explain transformer architecture.\nA: Transformers use self-attention to weigh token relationships regardless of distance. Encoder-decoder architecture: encoder builds context representations; decoder generates output. Positional encoding handles sequence order.\nTip: Mention BERT (encoder-only), GPT (decoder-only), T5 (both)."},

    # ════════════════════════════════════════════════════
    # PRODUCT MANAGER
    # ════════════════════════════════════════════════════
    {"id":"pm_001","role":"product_manager","category":"behavioral","level":"all","content":"Q: How do you prioritize a product backlog?\nA: Use RICE (Reach, Impact, Confidence, Effort), MoSCoW, or Kano model. Align with OKRs, gather customer feedback, balance technical debt vs new features.\nTip: Always tie prioritization to business metrics and user pain points."},
    {"id":"pm_002","role":"product_manager","category":"situational","level":"mid_senior","content":"Q: A key metric drops 20% overnight. Walk me through your investigation.\nA: Verify data pipeline, segment by platform/region, check recent deployments, A/B tests, external factors. Communicate findings with hypothesis-first approach.\nTip: Show structured thinking — interviewers value process over quick guesses."},
    {"id":"pm_003","role":"product_manager","category":"technical","level":"all","content":"Q: What is a North Star metric and why does it matter?\nA: A single metric that best captures the core value delivered to users — e.g., Weekly Active Users for Slack, Nights Booked for Airbnb. It aligns all teams toward a common goal.\nTip: Give an example from a product you've worked on or admire."},
    {"id":"pm_004","role":"product_manager","category":"behavioral","level":"all","content":"Q: Tell me about a product you launched from 0 to 1.\nA (STAR): Identified user pain via discovery interviews. Defined MVP scope. Led cross-functional team. Launched to 500 beta users, iterated on feedback, reached 10K MAU in 3 months.\nTip: Show the full product lifecycle — discovery, delivery, growth."},

    # ════════════════════════════════════════════════════
    # DEVOPS / CLOUD
    # ════════════════════════════════════════════════════
    {"id":"devops_001","role":"devops_engineer","category":"technical","level":"all","content":"Q: Explain blue-green vs canary deployments.\nA: Blue-green — two identical environments, switch traffic instantly, easy rollback. Canary — gradually shift traffic to new version, monitor, increase. Canary reduces blast radius.\nTip: Mention Argo Rollouts, AWS CodeDeploy, or Kubernetes traffic splitting."},
    {"id":"devops_002","role":"devops_engineer","category":"technical","level":"senior","content":"Q: How do you design a highly available Kubernetes cluster?\nA: Multi-zone control plane, pod disruption budgets, HPA, node auto-provisioner, multi-AZ PVCs, network policies, readiness/liveness probes, chaos engineering drills.\nTip: Reference IBM Cloud Kubernetes Service or OpenShift experience."},
    {"id":"devops_003","role":"devops_engineer","category":"technical","level":"all","content":"Q: What is Infrastructure as Code (IaC)? Which tools have you used?\nA: IaC manages infrastructure through code (Terraform, Ansible, CloudFormation). Benefits: versioning, reproducibility, drift detection, peer review via PRs.\nTip: Compare declarative (Terraform) vs imperative (Ansible) approaches."},
    {"id":"devops_004","role":"devops_engineer","category":"technical","level":"mid_senior","content":"Q: How do you implement zero-downtime deployments?\nA: Rolling updates, blue-green, canary, feature flags, health checks, readiness probes, database migration strategies (backward-compatible migrations, expand-contract pattern).\nTip: Discuss the database schema migration challenge — it's often overlooked."},
    {"id":"devops_005","role":"devops_engineer","category":"situational","level":"all","content":"Q: A container keeps crash-looping in production. How do you debug it?\nA: kubectl describe pod → check events; kubectl logs --previous → check crash reason; check resource limits (OOMKilled); verify config/secrets; check liveness probe config.\nTip: Walk through a real incident you debugged. Interviewers love specifics."},

    # ════════════════════════════════════════════════════
    # ML ENGINEER
    # ════════════════════════════════════════════════════
    {"id":"mle_001","role":"ml_engineer","category":"technical","level":"all","content":"Q: What is the difference between a Data Scientist and an ML Engineer?\nA: Data Scientists focus on analysis, modeling, and insights. ML Engineers productionize models: scalable serving, CI/CD for ML, monitoring for drift, feature stores, A/B testing infrastructure.\nTip: Show you understand both roles and can bridge the gap."},
    {"id":"mle_002","role":"ml_engineer","category":"technical","level":"mid_senior","content":"Q: How do you monitor ML models in production?\nA: Track data drift (statistical tests on input distributions), model drift (performance metrics over time), data quality checks, shadow mode testing, canary rollouts.\nTip: Mention tools: Evidently AI, Arize, MLflow, or IBM Watson OpenScale."},
    {"id":"mle_003","role":"ml_engineer","category":"technical","level":"senior","content":"Q: Explain the MLOps lifecycle.\nA: Data ingestion → Feature engineering → Model training → Evaluation → Registration → Deployment → Monitoring → Retraining. Automation via ML pipelines (Kubeflow, MLflow, SageMaker Pipelines).\nTip: Mention CI/CD for ML models and the key difference from software CI/CD."},

    # ════════════════════════════════════════════════════
    # FRONTEND DEVELOPER
    # ════════════════════════════════════════════════════
    {"id":"fe_001","role":"frontend_developer","category":"technical","level":"all","content":"Q: Explain the Virtual DOM and how React uses it.\nA: The Virtual DOM is a lightweight in-memory representation of the real DOM. React diffs new vs old Virtual DOM (reconciliation) and applies only necessary real DOM updates, improving performance.\nTip: Explain the reconciliation algorithm and React Fiber."},
    {"id":"fe_002","role":"frontend_developer","category":"technical","level":"all","content":"Q: What is the difference between `==` and `===` in JavaScript?\nA: `==` performs type coercion before comparing; `===` checks both value and type without coercion. Best practice: always use `===` to avoid unexpected bugs.\nTip: Give a tricky example: `0 == false` is true, `0 === false` is false."},
    {"id":"fe_003","role":"frontend_developer","category":"technical","level":"mid_senior","content":"Q: What are React hooks and why were they introduced?\nA: Hooks (useState, useEffect, useContext, etc.) allow functional components to use state and lifecycle features previously only available in class components. They improve code reuse and reduce boilerplate.\nTip: Explain rules of hooks and a custom hook you've written."},
    {"id":"fe_004","role":"frontend_developer","category":"technical","level":"senior","content":"Q: How do you optimize a React application's performance?\nA: Code splitting (React.lazy), memoization (useMemo, useCallback, React.memo), virtualization for long lists (react-window), bundle analysis, image optimization, CDN, avoiding unnecessary re-renders.\nTip: Show you profile first with React DevTools before optimizing."},

    # ════════════════════════════════════════════════════
    # BACKEND DEVELOPER
    # ════════════════════════════════════════════════════
    {"id":"be_001","role":"backend_developer","category":"technical","level":"all","content":"Q: What are the HTTP methods and when do you use them?\nA: GET (read), POST (create), PUT (replace), PATCH (partial update), DELETE (remove), OPTIONS (CORS preflight), HEAD (headers only). Use idempotency correctly.\nTip: Discuss idempotency — GET/PUT/DELETE are idempotent; POST is not."},
    {"id":"be_002","role":"backend_developer","category":"technical","level":"mid_senior","content":"Q: How do you implement authentication and authorization in an API?\nA: Authentication: JWT/OAuth 2.0/API keys. Authorization: RBAC or ABAC. Use HTTPS, short-lived tokens, refresh tokens, store secrets in vaults not code.\nTip: Discuss the difference between authentication (who you are) and authorization (what you can do)."},
    {"id":"be_003","role":"backend_developer","category":"technical","level":"senior","content":"Q: How do you design a rate limiter?\nA: Token bucket or leaky bucket algorithm. Store counts in Redis with TTL. Can be per-user, per-IP, or per-endpoint. Return 429 with Retry-After header.\nTip: Discuss distributed rate limiting challenges with multiple API nodes."},
    {"id":"be_004","role":"backend_developer","category":"technical","level":"all","content":"Q: What is connection pooling and why does it matter?\nA: Reusing pre-established database connections instead of creating new ones per request. Reduces latency and resource usage. Configure pool size based on DB max connections and application concurrency.\nTip: Mention the N+1 query problem as a related performance issue."},

    # ════════════════════════════════════════════════════
    # QA / TEST ENGINEER
    # ════════════════════════════════════════════════════
    {"id":"qa_001","role":"qa_engineer","category":"technical","level":"all","content":"Q: What is the testing pyramid?\nA: Unit tests (many, fast, cheap) at the base → Integration tests (middle) → E2E tests (few, slow, expensive) at the top. More unit tests catch bugs earlier and cheaper.\nTip: Mention why inverting the pyramid (too many E2E tests) is an anti-pattern."},
    {"id":"qa_002","role":"qa_engineer","category":"technical","level":"mid_senior","content":"Q: What is the difference between functional and non-functional testing?\nA: Functional: does the system do what it should (unit, integration, E2E)? Non-functional: performance testing, security testing, usability, scalability, reliability.\nTip: Give examples of non-functional requirements you've tested."},
    {"id":"qa_003","role":"qa_engineer","category":"technical","level":"all","content":"Q: Explain shift-left testing.\nA: Moving testing earlier in the SDLC — writing tests before or during development rather than after. Reduces cost of fixing bugs, encourages TDD/BDD, improves code quality.\nTip: Contrast with shift-right (production testing/monitoring)."},

    # ════════════════════════════════════════════════════
    # SYSTEM DESIGN (role-agnostic)
    # ════════════════════════════════════════════════════
    {"id":"sys_001","role":"general","category":"system_design","level":"senior","content":"Q: Design a notification system for 100M users.\nA: Producer service → Message queue (Kafka) → Consumer workers → Multi-channel delivery (FCM, APNS, email via SES). Use fan-out for broadcasts, user preferences table, rate limiting per user.\nTip: Discuss idempotency, delivery guarantees (at-least-once), and retry logic."},
    {"id":"sys_002","role":"general","category":"system_design","level":"senior","content":"Q: Design Twitter's trending topics feature.\nA: Aggregate tweet counts in sliding windows using Redis sorted sets or Kafka Streams. Dedup, filter spam/bots, rank by velocity not just volume. Geo-personalize.\nTip: Discuss approximate counting (Count-Min Sketch) vs exact counting."},
    {"id":"sys_003","role":"general","category":"system_design","level":"mid_senior","content":"Q: How would you cache data to improve API performance?\nA: Multi-level caching: in-process (local dict), distributed (Redis/Memcached), CDN (static assets). Cache-aside vs write-through vs write-behind patterns. Set TTLs, use cache invalidation strategies.\nTip: The hardest problems in caching: invalidation and cold start."},
    {"id":"sys_004","role":"general","category":"system_design","level":"senior","content":"Q: Design a real-time leaderboard.\nA: Redis Sorted Sets for O(log N) rank updates; persist to DB asynchronously. Shard by game/region for scale. Use WebSockets or SSE for real-time push to clients.\nTip: Discuss consistency trade-offs — eventual vs strong for leaderboard reads."},

    # ════════════════════════════════════════════════════
    # HR / BEHAVIORAL (universal)
    # ════════════════════════════════════════════════════
    {"id":"hr_001","role":"general","category":"hr","level":"all","content":"Q: Tell me about yourself.\nA: Use Present-Past-Future: current role and key achievements, relevant background that led here, and why this role excites you. Keep it under 2 minutes.\nTip: Tailor the narrative to skills mentioned in the job description."},
    {"id":"hr_002","role":"general","category":"hr","level":"all","content":"Q: What is your greatest weakness?\nA: Choose a real weakness, explain how you recognized it, and describe concrete improvement steps. Example: 'I used to avoid public speaking. I joined Toastmasters and now lead team demos.'\nTip: Show self-awareness + growth mindset. Never say 'I'm a perfectionist'."},
    {"id":"hr_003","role":"general","category":"hr","level":"all","content":"Q: Where do you see yourself in 5 years?\nA: Express ambition aligned with the company's growth path. Show you want to deepen expertise, take on leadership, and contribute to the company's mission.\nTip: Research the company's leadership development programs beforehand."},
    {"id":"hr_004","role":"general","category":"hr","level":"all","content":"Q: Why are you leaving your current job?\nA: Focus on what you're moving toward, not away from. 'I'm excited about the scale of challenges here / the tech stack / the mission.' Avoid badmouthing past employers.\nTip: Keep it brief and positive. Interviewers remember negativity."},
    {"id":"hr_005","role":"general","category":"hr","level":"all","content":"Q: Describe a conflict with a coworker and how you resolved it.\nA (STAR): Disagreed on architecture approach. Arranged 1:1, listened to their perspective, found common ground by proposing a POC to test both approaches. Result: adopted a hybrid solution, improved team trust.\nTip: Show emotional intelligence — listening and empathy matter more than being right."},
    {"id":"hr_006","role":"general","category":"hr","level":"all","content":"Q: Why should we hire you?\nA: I bring [specific skill 1], [specific skill 2], and [specific achievement] that directly maps to [role requirement]. I am also a fast learner and a collaborative team player.\nTip: This is your 60-second pitch. Prepare it word-for-word in advance."},
    {"id":"hr_007","role":"general","category":"hr","level":"all","content":"Q: What motivates you?\nA: Be specific: solving hard technical problems, seeing user impact, continuous learning, or building scalable systems. Tie your motivation to this company's work.\nTip: Vague answers ('I like challenges') lose points. Give a concrete story."},

    # ════════════════════════════════════════════════════
    # BEHAVIORAL (STAR framework)
    # ════════════════════════════════════════════════════
    {"id":"beh_001","role":"general","category":"behavioral","level":"all","content":"Q: Tell me about a time you failed and what you learned.\nA (STAR): Deployed untested code to production, caused 2-hour outage. Owned the mistake immediately, led post-mortem, introduced mandatory code review and staging environment policy. Never happened again.\nTip: Owning failure and showing systemic improvement is more impressive than claiming you've never failed."},
    {"id":"beh_002","role":"general","category":"behavioral","level":"all","content":"Q: Describe a time you worked under tight deadlines.\nA (STAR): 3-day sprint to deliver a client demo. Ruthlessly prioritized MVP features, cut scope with stakeholder agreement, delivered on time. Client converted to paid plan.\nTip: Show prioritization and communication skills, not just hustle."},
    {"id":"beh_003","role":"general","category":"behavioral","level":"mid_senior","content":"Q: Tell me about a time you influenced a decision without authority.\nA (STAR): Convinced team to adopt code reviews by showing data on bug rates and sharing an industry study. Built consensus through data and demos, not mandates.\nTip: Leadership without authority is a key skill for senior roles."},
    {"id":"beh_004","role":"general","category":"behavioral","level":"all","content":"Q: Describe a time you went above and beyond.\nA (STAR): Noticed a recurring customer complaint pattern, built an automated monitoring dashboard in personal time, presented to management. Result: 30% reduction in support tickets.\nTip: Pick a story that shows initiative and business impact, not just extra hours."},

    # ════════════════════════════════════════════════════
    # NEGOTIATION & OFFER
    # ════════════════════════════════════════════════════
    {"id":"neg_001","role":"general","category":"negotiation","level":"all","content":"Strategy: Salary Negotiation Best Practices\n1) Research market rates (Glassdoor, Levels.fyi, LinkedIn Salary). 2) Let employer offer first. 3) Counter with a specific number. 4) Negotiate total comp: base, bonus, equity, PTO, remote.\n5) Get offer in writing before resigning.\nOpener: 'I'm excited about this offer. Based on my research, I was hoping for X. Is there flexibility?'"},

    # ════════════════════════════════════════════════════
    # ENTRY LEVEL / FRESHER
    # ════════════════════════════════════════════════════
    {"id":"entry_001","role":"general","category":"entry_level","level":"fresher","content":"Q: You have no work experience. Why should we hire you?\nA: Highlight academic projects, internships, open-source contributions, certifications, and soft skills. Demonstrate eagerness to learn, cultural fit, and fresh perspective.\nTip: Quantify project outcomes — users, accuracy, performance gains."},
    {"id":"entry_002","role":"general","category":"entry_level","level":"fresher","content":"Q: What did you build in your final year project?\nA: Describe the problem, technology stack, your specific contribution, and the outcome. Use numbers wherever possible.\nTip: Prepare a 2-minute crisp demo walkthrough — even if they don't ask to see it."},

    # ════════════════════════════════════════════════════
    # IBM / COMPANY SPECIFIC
    # ════════════════════════════════════════════════════
    {"id":"ibm_001","role":"general","category":"company_specific","level":"all","content":"Q: Why do you want to work at IBM?\nA: Reference IBM's leadership in hybrid cloud, AI (watsonx), quantum computing, and consulting. Mention specific IBM products or research. Connect career goals to IBM's mission.\nTip: Mention IBM SkillsBuild, IBM Research, or a specific IBM patent/paper you admire."},
    {"id":"ibm_002","role":"general","category":"company_specific","level":"all","content":"Q: What do you know about IBM watsonx?\nA: watsonx.ai is IBM's enterprise AI platform with IBM Granite foundation models for text, code, and geospatial data. watsonx.data is a data lakehouse. watsonx.governance provides AI governance.\nTip: Mention how it differs from OpenAI — enterprise focus, trust, explainability, on-premise deployment."},

    # ════════════════════════════════════════════════════
    # MOCK INTERVIEW TIPS
    # ════════════════════════════════════════════════════
    {"id":"tip_001","role":"general","category":"hr","level":"all","content":"Mock Interview Tips:\n1) Use the STAR method for behavioral questions (Situation, Task, Action, Result).\n2) Pause before answering — it shows thoughtfulness, not hesitation.\n3) Think aloud during technical problems — interviewers assess your process.\n4) Ask clarifying questions before solving system design problems.\n5) End every answer with a confident summary sentence."},
    {"id":"tip_002","role":"general","category":"hr","level":"all","content":"Communication Tips for Interviews:\n1) Speak at 70% of your normal speed — adrenaline speeds you up.\n2) Use concrete examples for every claim about yourself.\n3) Quantify impact: 'reduced latency by 40%' beats 'improved performance'.\n4) Mirror the interviewer's vocabulary — use their exact terms.\n5) Maintain eye contact; nod to show engagement."},
]


class RAGPipeline:
    """Manages vector store indexing and semantic retrieval for interview content."""

    def __init__(self) -> None:
        cfg = get_settings()
        os.makedirs(cfg.chroma_db_path, exist_ok=True)

        self._chroma = chromadb.PersistentClient(
            path=cfg.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._chroma.get_or_create_collection(
            name=cfg.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._cfg = cfg

        if self._collection.count() < len(SEED_DOCUMENTS):
            self._seed_knowledge_base()

    def _seed_knowledge_base(self) -> None:
        logger.info("Seeding knowledge base with %d documents…", len(SEED_DOCUMENTS))
        existing_ids = set(self._collection.get(include=[])["ids"])
        new_docs = [d for d in SEED_DOCUMENTS if d["id"] not in existing_ids]
        if not new_docs:
            return
        ids = [doc["id"] for doc in new_docs]
        texts = [doc["content"] for doc in new_docs]
        metadatas = [{"role": d["role"], "category": d["category"], "level": d["level"]} for d in new_docs]
        embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()
        self._collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        logger.info("Knowledge base ready: %d total documents.", self._collection.count())

    def ingest_text(self, text: str, metadata: dict[str, str] | None = None) -> str:
        cfg = self._cfg
        chunks = self._chunk_text(text, cfg.rag_chunk_size, cfg.rag_chunk_overlap)
        ids = [f"upload_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]
        embeddings = self._embedder.encode(chunks, show_progress_bar=False).tolist()
        meta = metadata or {}
        self._collection.add(ids=ids, documents=chunks, metadatas=[meta]*len(chunks), embeddings=embeddings)
        logger.info("Ingested %d chunks.", len(chunks))
        return f"Indexed {len(chunks)} chunks into the knowledge base."

    def retrieve(self, query: str, role: str | None = None, top_k: int | None = None) -> list[str]:
        k = top_k or self._cfg.rag_top_k_results
        query_embedding = self._embedder.encode([query], show_progress_bar=False).tolist()[0]
        where_filter: dict | None = None
        if role and role not in ("general", ""):
            where_filter = {"$or": [{"role": role}, {"role": "general"}]}
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
            where=where_filter,
            include=["documents"],
        )
        return results["documents"][0] if results["documents"] else []

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        words = text.split()
        if not words:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += chunk_size - overlap
        return chunks


_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
